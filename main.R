library(dplyr)

# import raw file
data_raw = read.csv('raw_data.csv')
data_raw
str(data_raw)

# check NAs
sum(is.na(data_raw[1:20]))

colSums(is.na(data_raw))

# amount of NA values in percent each columns
missing_percentage <- colSums(is.na(data_raw)) / nrow(data_raw) * 100
missing_percentage

# check if there is a space after any value
sapply(data_raw, function(x) any(grepl("^\\s+|\\s+$", x)))


########## clean up ########## 

data <- data_raw

# convert character to factor data type
data <- data %>%
  mutate(across(where(is.character), as.factor))
data

# mutate signup_actual_date and days_since_signup
data <- data %>%
  mutate(
    signup_actual_date = as.Date(Sys.Date()) + signup_date,
    days_since_signup = as.numeric(Sys.Date() - signup_actual_date))


# median imputation
data <- data %>%
  mutate(
    age = ifelse(is.na(age), median(age, na.rm = TRUE), age),
    weekly_hours = ifelse(is.na(weekly_hours), median(weekly_hours, na.rm = TRUE), weekly_hours))


write.csv(data,"cleaned_data.csv")

