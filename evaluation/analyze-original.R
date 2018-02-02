library(DBI)
library(ggplot2)
library(plyr)
library(psych)
library(xtable)

# Select the training set size:
#  small set      -- 32
#  medium set     -- 512
#  original set   -- 11000
TRAINING_SET_SIZE <- 11000


# Calculate the reciprocal of the rank. Nulls are converted to zeros.
reciprank <- function (vec) {
  rrs <- 1.0 / vec
  rrs[is.na(rrs)] = 0
  return(rrs)
}


# Fetch the raw results from the database.
con <- dbConnect(RSQLite::SQLite(), "results.sqlite3")
raw.results <- dbGetQuery(con, "
  SELECT line_location_rank, exact_location_rank, valid_fix_rank, true_fix_rank,
           forwards_val_loss, backwards_val_loss,
         hidden_layers, context_length,
           patience, partition,
         training_set_size,
        bad_file
    FROM result JOIN model ON result.model_id = model.id;
")
dbDisconnect(con)

# Apply some transformations to the raw results.
results <- within(raw.results, {
  # Treat these as factors rather than continuous variables
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

# Select a subset according to the desired training set size.
if (!is.na(TRAINING_SET_SIZE)) {
  results <- subset(results, training_set_size==TRAINING_SET_SIZE)
}

# Get the MRR for all these responding variables.
aggdata.partitioned <- with(results, {aggregate(
  # These are the responding variables that should be meaned.
  cbind(line_location_rr, exact_location_rr, valid_fix_rr, true_fix_rr, mean_val_loss) ~
  # These are the manipulated variables; there may be more!
    hidden_layers + context_length + patience + partition,
  results, mean
)})

aggdata <- with(results, {aggregate(
  # These are the responding variables that should be meaned.
  cbind(line_location_rr, exact_location_rr, valid_fix_rr, true_fix_rr, mean_val_loss) ~
    # These are the manipulated variables; there may be more!
    hidden_layers + context_length + patience,
  results, mean
)})

MAX_RANK <- 5

results.300 <- subset(results, hidden_layers == 300)
results.300 <- within(results.300, {
  true_fix_rank[is.na(true_fix_rank)] <- Inf
  true_fix_rank[true_fix_rank >= MAX_RANK] <- MAX_RANK
  # Convert into an ordinal factor
  true_fix_rank <- ordered(true_fix_rank,
                           levels=1:MAX_RANK,
                           labels=c("1", "2", "3", "4", "5+"))
})

amount.first <- length(which(results.300$true_fix_rank == 1))
pct.first <- 100 * amount.first / nrow(results.300)

ggplot(results.300, aes(x=true_fix_rank)) +
  geom_bar()

#ggplot(results.300, aes(x=hidden_layers, y=valid_fix_rr)) + geom_violin()
#ggplot(results.300, aes(x=hidden_layers, y=exact_location_rr)) + geom_violin()


mean(results.300$line_location_rr)
mean(results.300$exact_location_rr)
mean(results.300$valid_fix_rr)
mean(results.300$true_fix_rr)


