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

# Calculate the reciprocal of the rank. Nulls are converted to zeros.
reciprank <- function (vec) {
  rrs <- 1.0 / vec
  rrs[is.na(rrs)] = 0
  return(rrs)
}

# Apply some transformations to the raw results.
results <- within(results, {
  # Treat these as factors rather than continuous variables
  dropout <- as.factor(dropout)
  patience <- as.factor(patience)
  partition <- as.factor(partition)
  
  # Get reciprocal ranks.
  line_location_rr <- reciprank(line_location_rank)
  exact_location_rr <- reciprank(exact_location_rank)
  valid_fix_rr <- reciprank(valid_fix_rank)
  true_fix_rr <- reciprank(true_fix_rank)
  
  # Figure out the mean validation loss
  mean_val_loss <- (forwards_val_loss + backwards_val_loss)/2
})

# Get the MRR for all these responding variables.
aggdata <- with(results, {aggregate(
  # These are the responding variables that should be meaned.
  cbind(line_location_rr, exact_location_rr, valid_fix_rr, true_fix_rr, mean_val_loss) ~
  # These are the manipulated variables; there may be more!
    hidden_layers + context_length + dropout + patience + optimizer + partition,
  results, mean
)})

# Figure out what actually affects line location MRR
line.model <- with(results, {lm(
  valid_fix_rr ~
    hidden_layers + context_length + dropout + patience + optimizer + partition + mean_val_loss
)})
summary(line.model)

# Is exact location correlated with the validation loss?
with(aggdata, cor.test(mean_val_loss, exact_location_rr))
# How about true fix?
with(aggdata, cor.test(mean_val_loss, true_fix_rr))

# Sample violin plot:
ggplot(results, aes(x=optimizer, y=exact_location_rr)) + geom_violin()
