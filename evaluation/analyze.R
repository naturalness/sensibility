library(DBI)
library(ggplot2)

# Select the training set size:
#  small set      -- 32
#  medium set     -- 512
#  larget set     -- 8192
#  huge set       -- 32768
#  enormous set   -- 131072
TRAINING_SET_SIZE <- 32


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
           IFNULL(CAST(dropout AS TEXT), 'None') as [dropout],
           patience, optimizer, partition,
         training_set_size
    FROM result JOIN model ON result.model_id = model.id;
")
dbDisconnect(con)

# Apply some transformations to the raw results.
results <- within(raw.results, {
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

# Select a subset according to the desired training set size.
if (!is.na(TRAINING_SET_SIZE)) {
  results <- subset(results, training_set_size==TRAINING_SET_SIZE)
}

# Get the MRR for all these responding variables.
aggdata <- with(results, {aggregate(
  # These are the responding variables that should be meaned.
  cbind(line_location_rr, exact_location_rr, valid_fix_rr, true_fix_rr, mean_val_loss) ~
  # These are the manipulated variables; there may be more!
    hidden_layers + context_length + dropout + patience + optimizer,
  results, mean
)})

# Figure out what actually affects line location MRR
line.model <- with(results, {lm(
  valid_fix_rr ~
    hidden_layers + context_length + dropout + patience + optimizer + mean_val_loss
)})
summary(line.model)

# Is exact location correlated with the validation loss?
with(aggdata, cor.test(mean_val_loss, exact_location_rr))
# How about true fix?
with(aggdata, cor.test(mean_val_loss, true_fix_rr))

wilcox.test(subset(aggdata, optimizer == "adam")$valid_fix_rr,
            subset(aggdata, optimizer == "rmsprop")$valid_fix_rr,
            alternative = "greater")

# Sample violin plot:
ggplot(results, aes(x=optimizer, y=exact_location_rr)) + geom_violin()

# How often is Adam better than RMSprop, controlling for all other manipulated variables?
library(plyr)
ddply(aggdata, .(hidden_layers, context_length, dropout, patience),
      function (df)
        df[df$optimizer=="adam",]$valid_fix_rr > df[df$optimizer=="rmsprop",]$valid_fix_rr
)
# By how much?
median(ddply(aggdata, .(hidden_layers, context_length, dropout, patience),
             function (df)
               df[df$optimizer=="adam",]$valid_fix_rr - df[df$optimizer=="rmsprop",]$valid_fix_rr)$V1
)

# How about dropout?
dropout.effect <- ddply(aggdata, .(hidden_layers, context_length, optimizer, patience),
      function (df)
        df[df$dropout=="0.75",]$valid_fix_rr < df[df$dropout=="None",]$valid_fix_rr
)
sum(dropout.effect$V1) / nrow(dropout.effect)

median(ddply(aggdata, .(hidden_layers, context_length, optimizer, patience),
             function (df)
               df[df$dropout=="None",]$valid_fix_rr - df[df$dropout=="0.75",]$valid_fix_rr)$V1
)
