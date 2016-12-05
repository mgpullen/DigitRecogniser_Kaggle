####################
# 
# Randomly shuffles the rows of the input data frame
# Inputs:  df - dplyr data frame to be shuffled
# 
# Outputs: df - the shuffled dplyr data frame
# 
####################

randShuffle <- function(df){
  # Randomly shuffle the data frame using the 'sample' function
  df <- df[sample(nrow(df)), ]
  return(df)
}