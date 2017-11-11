library(DBI)
library(ggplot2)

con <- dbConnect(RSQLite::SQLite(), "results.sqlite3")

results <- dbGetQuery(con, "
  SELECT line_location_rank, exact_location_rank, valid_fix_rank, true_fix_rank,
         hidden_layers, context_length,
           IFNULL(CAST(dropout AS TEXT), 'None') as [dropout],
           patience, optimizer
    FROM result JOIN model ON result.model_id = model.id;
")

attach(results)

# Treat these as factors rather than continuous variables
results$dropout <- as.factor(dropout)
results$patience <- as.factor(patience)


reciprank <- function (vec) {
  rrs <- 1.0 / vec
  rrs[is.na(rrs)] = 0
  return(rrs)
}

# Get reciprocal ranks.
results$line_location_rr <- reciprank(line_location_rank)
results$exact_location_rr <- reciprank(exact_location_rank)
results$valid_fix_rr <- reciprank(valid_fix_rank)
results$true_fix_rr <- reciprank(true_fix_rank)

# Attach to new columns without invoking warnings:
detach(results)
attach(results)

# Get the MRR for all these responding variables.
aggdata <- aggregate(
  # These are the means I want
  cbind(line_location_rr, exact_location_rr, valid_fix_rr, true_fix_rr) ~
  # These are the responding variables; there may be more!
    hidden_layers + context_length + dropout + patience + optimizer,
  results, mean
)

# Figure out what actually affects line location MRR
line.model <- lm(line_location_rr ~ hidden_layers + context_length + dropout + patience + optimizer)

ggplot(results, aes(x=patience, y=line_location_rr)) + geom_violin()