####################
# 
# Randomly shuffles the rows of the input data frame
# Inputs:  df - data frame to be shuffled
# 
# Outputs: df - the shuffled data frame
# 
####################

randShuffle <- function(df){
  # Randomly shuffle the data frame using the 'sample' function
  df <- df[sample(nrow(df)), ]
  return(df)
}
