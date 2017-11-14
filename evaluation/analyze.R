library(DBI)
library(ggplot2)

con <- dbConnect(RSQLite::SQLite(), "results.sqlite3")
results <- dbGetQuery(con, "
  SELECT line_location_rank, exact_location_rank, valid_fix_rank, true_fix_rank,
           forwards_val_loss, backwards_val_loss,
         hidden_layers, context_length,
           IFNULL(CAST(dropout AS TEXT), 'None') as [dropout],
           patience, optimizer, partition
    FROM result JOIN model ON result.model_id = model.id;
")
dbDisconnect(con)

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
# Can't use with()/within() because I want to create variables outside
# of the environment
detach(results)
attach(results)

# Get the MRR for all these responding variables.
aggdata <- aggregate(
  # These are the responding variables that should be meaned.
  cbind(line_location_rr, exact_location_rr, valid_fix_rr, true_fix_rr,
        forwards_val_loss, backwards_val_loss) ~
  # These are the manipulated variables; there may be more!
    hidden_layers + context_length + dropout + patience + optimizer + partition,
  results, mean
)

# Figure out what actually affects line location MRR
line.model <- lm(exact_location_rr ~
                   hidden_layers + context_length + dropout + patience + optimizer + partition)

detach(results)

# Is exact location correlated with the validation loss?
with(aggdata, cor((forwards_val_loss + backwards_val_loss) / 2, exact_location_rr))

# Sample violin plot:
ggplot(results, aes(x=optimizer, y=line_location_rr)) + geom_violin
