library(DBI)
library(ggplot2)
library(plyr)
library(psych)
library(xtable)

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


# How often is Adam better than RMSprop, controlling for all other manipulated variables?
optimizer.effect <- ddply(aggdata, .(hidden_layers, context_length, dropout, patience),
  function (df) {
    adam <- df[df$optimizer=="adam",]
    rmsprop <- df[df$optimizer=="rmsprop",]

    return(data.frame(valid.greater=adam$valid_fix_rr > rmsprop$valid_fix_rr,
                      valid.difference=adam$valid_fix_rr - rmsprop$valid_fix_rr,
                      true.greater=adam$true_fix_rr > rmsprop$true_fix_rr,
                      true.difference=adam$true_fix_rr - rmsprop$true_fix_rr))
  }
)
list(greater=sum(optimizer.effect$true.greater), out.of=nrow(optimizer.effect))
t.test(optimizer.effect$true.difference)
median(optimizer.effect$true.difference)

# What is the effect of applying naïve dropout?
dropout.effect <- ddply(aggdata, .(hidden_layers, context_length, optimizer, patience),
  function (df) {
    none <- df[df$dropout=="None",]
    dropout <- df[df$dropout=="0.75",]

    return(data.frame(valid.greater=none$valid_fix_rr > dropout$valid_fix_rr,
                      valid.difference=none$valid_fix_rr - dropout$valid_fix_rr,
                      true.greater=none$true_fix_rr > dropout$true_fix_rr,
                      true.difference=none$true_fix_rr - dropout$true_fix_rr))
  }
)
list(greater=sum(dropout.effect$true.greater), out.of=nrow(dropout.effect))
t.test(dropout.effect$true.difference)
median(dropout.effect$true.difference)


# Sample violin plot:
ggplot(results, aes(x=optimizer, y=true_fix_rr)) + geom_violin()

# Create the contents of tab:top-configs
columns <- c("hidden_layers", "context_length", "dropout", "patience",
             "optimizer", "exact_location_rr", "valid_fix_rr", "true_fix_rr")
print(xtable(subset(aggdata, select=columns)[c(93, 18, 88),]),
      only.contents = TRUE,
      booktabs = TRUE,
      include.rownames = FALSE)
