#Capstone Project
#====== Package load ======
require(MASS)
require(e1071)
require(rpart)
require(rpart.plot)
require(randomForest)
require(gbm)
require(pls)
require(neuralnet)
require(nnet)
require(gam)
require(leaps)
require(dummies)
require(car)
require(glmnet)
require(FNN)
require(caret)
require(ROCR)
require(psych)
require(forecast)
require(grid)
require(caret)
require(xgboost)
require(ggplot2)
require(gridExtra)
require(reshape2)
require(plyr)
require(dplyr)
source('https://raw.githubusercontent.com/jchristo12/general/master/r_udf.R')
#source('C:/Users/Joe/general-code/r_udf.R')

#======= Data Load and Cleanup======
#change the working directory
setwd("C:/Users/Joe/OneDrive - Northwestern University - Student Advantage/498 - Capstone Project")

#set the strings that will be read as 'NA'
nas = c("", '#NA', "NA", "N.A.", "#N/A")
#path for the data
data_path <- 'C:/Users/Joe/OneDrive - Northwestern University - Student Advantage/498 - Capstone Project/financial-analysis-model/'
github <- 'https://raw.githubusercontent.com/jchristo12/MSDSCapstone/master/'
#read in the data
df <- read.csv(paste0(github, 'financial-analysis-data.csv'), na.strings=nas)
#drop the performance rating fields
df <- df[,-c(27:28)]


#exclude New York from the analysis
#df <- df %>% filter(city != 'New York')

#convert the columns into the correct data types
df$team.champs.5yr <- as.factor(df$team.champs.5yr)
#df$city.franchises <- as.factor(df$city.franchises)


#find features that have 0s that shouldn't
missing_values(df)
df %>%
  select(-c(1:7,12:14,18,26)) %>%
  na_if(0) %>%
  missing_values()
#team.value and team-ticket have a couple that should be NAs
df$team.value <- ifelse(df$team.value == 0, NA, df$team.value)
df$team.ticket <- ifelse(df$team.ticket == 0, NA, df$team.ticket)


#read in player stats dataframe
player <- read.csv(paste0(github, 'player-stats.csv'))
player <- player[,c('Year', 'Player', 'Tm', 'WS')]
player$team.year <- as.factor(paste0(player$Tm, "-", player$Year))
player <- player[!is.na(player$Year) & player$Year >= 2004 & player$Tm != '',]
player <- player[, c(2,4:5)]
WS_thres <- 7
player$superstar <- as.factor(ifelse(player$WS >= WS_thres, 'Y', 'N'))

player_agg <- player %>%
  group_by(team.year) %>%
  filter(superstar == 'Y') %>%
  summarize('team.superstar'=n())

df <- left_join(df, player_agg, by='team.year')
df$team.superstar <- ifelse(is.na(df$team.superstar), 0, df$team.superstar)


#read in team stats data
stats_df_combo <- function(path, na_string, orig_df){
  stats <- read.csv(path, na.strings=na_string)
  #derive variables that only require the stats data frame
  stats$pts.per.gm <- stats$PTS / stats$G
  new_df <- left_join(orig_df, stats, by='team.year')
  new_df <- subset(new_df, select=-c(Season, Team, team.abbr))
  #derive variables that require both data frames
  new_df$salary.per.pt <- new_df$team.salary / new_df$PTS
  return(new_df)
}
#df <- stats_df_combo(paste0(data_path, 'team-stats.csv), nas, df)  don't want to include this dataframe


#create the data partition
set.seed(6)
trainIndex <- createDataPartition(df$team.revenue, p=0.70, list=FALSE)
df.train <- df[trainIndex,]
df.test <- df[-trainIndex,]


#====== Data Exploration ======
#check for missing values
missing_values(df.train)

#analyze team revenue
ggplot(df.train) +
  geom_histogram(aes(team.revenue), bins=30) +
  labs(title='Annual Franchise Revenue', x='Revenue ($m)', y='Count')
df.train[df.train$team.revenue > 250,]
#   transformed target variable
ggplot(df.train) +
  geom_point(aes(x=log(city.returns), y=log(team.revenue)))

ggplot(df.train) +
  geom_point(aes(x=team.fci, y=team.revenue, color=team.champs.5yr))


#team value
ggplot(df.train) +
  geom_histogram(aes(team.value), bins=30)

#number of championships in last 5yrs
ggplot(df.train) +
  geom_boxplot(aes(x=team.champs.5yr, y=team.revenue))



#impute variables using decision trees
df.train.impute <- df.train %>%
  select(-c(1:3,5:7))
impute_vars <- names(df.train.impute)[-c(1, 9)]
df.train.imp <- impute_trees(df.train.impute, impute_vars)
#store the training fits to use on the test data
impute.fits <- store_impute_fit(df.train.impute, impute_vars)
#rm(df.train.impute)
#combine the imputed variables to the main training dataframe
df.train.imp <- cbind(df.train[,c(1:3,5:7)], df.train.imp) %>% data.frame()

missing_values(df.train.imp)


#create derived features
df.train.imp$city.unemploy.rate <- df.train.imp$imp_city.unemployed / df.train.imp$imp_city.work.force
df.train.imp$team.avg.attend <- df.train.imp$imp_team.total.attend / df.train.imp$team.total.gms
df.train.imp$team.salary.per.win <- df.train.imp$imp_team.salary / df.train.imp$imp_team.wins
df.train.imp$team.revenue.multiple <- df.train.imp$imp_team.value / df.train.imp$imp_team.revenue
df.train.imp$team.salary.per.attend <- df.train.imp$imp_team.salary / df.train.imp$imp_team.total.attend
df.train.imp$team.attend.revenue <- df.train.imp$imp_team.total.attend * df.train.imp$imp_team.ticket
#df.train.imp$city.salary.per.capita <- df.train.imp$imp_city.salary / df.train.imp$imp_city.returns

#transform variables
df.train.imp$trans_team.champs.5yr <- ifelse(df.train.imp$imp_team.champs.5yr != '0',  'Y', 'N') %>% factor()
#df.train.imp$trans_city.returns <- log(df.train.imp$imp_city.returns)
#df.train.imp$trans_city.exempt <- log(df.train.imp$imp_city.exempt)
df.train.imp$team.superstar.cat <- as.factor(df.train.imp$imp_team.superstar)


#analyze the categorical superstar on team value or revenue
ggplot(df.train.imp) +
  geom_boxplot(aes(x=team.superstar.cat, y=team.value))



#grahpical exploration of derived and transformed features
p2 <- ggplot(df.train.imp)

p2 + geom_point(aes(x=team.salary.per.win, y=imp_team.revenue, color=city.franchises))
p2 + geom_point(aes(x=team.attend.revenue, y=imp_team.revenue))


#create a correlation headmap
df.train.imp %>%
  select_if(is.numeric) %>%
  select(-c(impute_vars, year, team.total.gms)) %>%
  cor_heatmap()

#====== Process the Test Data ======
#impute test data with training fit function
impute_test_local <- function(obj.list, test.df, cols){
  for(i in cols){
    n <- which(cols == i)
    if(is.factor(test.df[[i]])){
      imp <- factor(ifelse(is.na(test.df[[i]]), predict(obj.list[[n]], test.df, type="class"), test.df[[i]]))
    } else{
      imp <- ifelse(is.na(test.df[[i]]), predict(obj.list[[n]], test.df), test.df[[i]])
    }
    test.df[[paste0("imp_",i)]] <- imp
  }
  return(test.df)
}

#impute missing data
df.test.imp <- impute_test_local(impute.fits, df.test, impute_vars)

#create derived features
df.test.imp$city.unemploy.rate <- df.test.imp$imp_city.unemployed / df.test.imp$imp_city.work.force
df.test.imp$team.avg.attend <- df.test.imp$imp_team.total.attend / df.test.imp$team.total.gms
df.test.imp$team.salary.per.win <- df.test.imp$imp_team.salary / df.test.imp$imp_team.wins
df.test.imp$team.revenue.multiple <- df.test.imp$imp_team.value / df.test.imp$imp_team.revenue
df.test.imp$team.salary.per.attend <- df.test.imp$imp_team.salary / df.test.imp$imp_team.total.attend
df.test.imp$team.attend.revenue <- df.test.imp$team.total.attend * df.test.imp$imp_team.ticket
#df.test.imp$city.salary.per.capita <- df.test.imp$imp_city.salary / df.test.imp$imp_city.returns

#transform variables
df.test.imp$trans_team.champs.5yr <- ifelse(df.test.imp$team.champs.5yr != '0',  'Y', 'N') %>% factor()
df.test.imp$trans_city.returns <- log(df.test.imp$imp_city.returns)
df.test.imp$trans_city.exempt <- log(df.test.imp$imp_city.exempt)



#====== Data Modeling ======
#set the training parameters
fitCtrl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- 'RMSE'

#universal unneeded variables for modeling
droplist <- c('city', 'team', 'nickname', 'city.year', 'team.year', 'nickname.year', 'imp_team.value', 'team.total.gms', 'team.revenue.multiple')
orig_var <- c(impute_vars, 'imp_team.champs.5yr')
other_drop_vars <- c('team.attend.revenue', 'team.superstar.cat')


#Linear Regression
#
#drop unneeded features
drop.sub1 <- c('year', 'imp_sp.return', 'imp_real.gdp.delta', 'imp_team.total.attend', 'imp_city.salary', 'imp_city.agi', 'imp_city.employed',
               'imp_city.work.force', 'imp_city.pop', 'team.salary.per.win', 'team.salary.per.attend', 'imp_city.unemployed', 'imp_team.fci',
               'imp_city.returns', 'imp_team.superstar')

subset1 <- create_subset(df.train.imp, c(droplist, orig_var, other_drop_vars, drop.sub1))
subset1 %>%
  select_if(is.numeric) %>%
  cor_heatmap()
#build the model
lin.mod.1 <- lm(log(imp_team.revenue)~., data=subset1)

summary(lin.mod.1)
par(mfrow=c(2,2))
plot(lin.mod.1)
par(mfrow=c(1,1))
hist(lin.mod.1$residuals)

cbind('actual'=subset1$imp_team.revenue, 'pred'=exp(lin.mod.1$fitted.values)) %>%
  data.frame() %>%
  ggplot() +
  geom_point(aes(x=actual, y=pred)) +
  geom_abline(slope=1, intercept=0, color='red') +
  labs(title='Actual vs. Fit', x='Actual', y='Fitted')

#estimate OOS error
lin.mod.1.pred <- predict(lin.mod.1, newdata=df.test.imp, interval='predict') %>%
  exp() %>%
  cbind('act'=df.test.imp$imp_team.revenue)
MAE(df.test.imp$imp_team.revenue, lin.mod.1.pred[,1])
RMSE(df.test.imp$imp_team.revenue, lin.mod.1.pred[,1])


#LASSO regression
drop_lasso <- c('year', 'imp_city.agi', 'imp_city.exempt')

subset2 <- create_subset(df.train.imp, c(droplist, orig_var, other_drop_vars, drop_lasso))
x.lasso <- model.matrix(imp_team.revenue~., data=subset2)[,-1]
y.lasso <- subset2$imp_team.revenue
#find the best lambda
set.seed(68)
cv.lasso <- cv.glmnet(x.lasso, y.lasso, alpha=1, nfolds=10)
bestLambda <- cv.lasso$lambda.min
lambda1se <- cv.lasso$lambda.1se
#create model
lasso.1 <- glmnet(x.lasso, y.lasso, alpha=1)
#take a look at coefficients
lasso.1.coef <- predict(lasso.1, type='coefficients', s=bestLambda)
lasso.2.coef <- predict(lasso.1, type='coefficients', s=lambda1se)
lasso.fit <- predict(lasso.1, s=lambda1se, newx=x.lasso)

cbind('actual'=y.lasso, 'fit'=c(lasso.fit)) %>%
  data.frame() %>%
  ggplot() +
  geom_point(aes(x=actual, y=fit)) +
  geom_abline(slope=1, intercept=0, color='red') +
  labs(title='Actual vs. Fit', subtitle='Training Data', x='Actual', y='Fitted')

#estimate OOS error
subtest2 <- create_subset(df.test.imp, c(droplist, orig_var, other_drop_vars, drop_lasso))
x.test.lasso <- model.matrix(imp_team.revenue~., subtest2)[,-1]
y.test.lasso <- subtest2$imp_team.revenue
lasso.1.pred <- predict(lasso.1, s=bestLambda, newx=x.test.lasso)
MAE(y.test.lasso, lasso.1.pred)
RMSE(y.test.lasso, lasso.1.pred)


#Random Forest
drop.sub3 <- c('year', 'imp_team.fci', 'imp_city.salary', 'trans_city.exempt', 'trans_city.returns', 'team.total.attend')
subset3 <- create_subset(df.train.imp, c(droplist, orig_var, other_drop_vars, drop.sub3))
names(subset3)
var_floor <- sqrt(ncol(subset3)-1) %>% floor()
grid <- expand.grid(.mtry=1:20)
rf_cv <- train(imp_team.revenue~., data=subset3, method='rf', metric=metric, trControl=fitCtrl, tuneGrid=grid, importance=TRUE)
rf_cv_final <- rf_cv$finalModel

rf <- randomForest(imp_team.revenue~., data=subset3, ntree=1000, mtry=10, importance=TRUE)
varImp(rf_cv)
varImpPlot(rf, main="Variable Importance Plot")

#OOS (rf) error estimation
rf.pred <- predict(rf, newdata=df.test.imp)
MAE(df.test.imp$imp_team.revenue, rf.pred)
RMSE(df.test.imp$imp_team.revenue, rf.pred)

#OOS (rf_cv) error estimation
rf_cv_pred <- predict(rf_cv_final, newdata=df.test.imp)


#XGBoost
subset4 <- create_subset(df.train.imp, c(droplist, orig_var, other_drop_vars))
#perform one hot encoding
#subset4 <- one_hot_encode(subset4)
#remove year from the dataframe
subset4 <- subset(subset4, select=-year)

#built training matrices
train_x <- model.matrix(imp_team.revenue~., data=subset4)[, -1]
train_y <- subset4$imp_team.revenue
dtrain <- xgb.DMatrix(data=train_x, label=train_y)

#set the tuning grid
tune_xgb <- expand.grid(nrounds=c(100,200,500), lambda=c(0.01, 0.1), alpha=1, eta=c(0.05))
#tune the xgboost algorithm
set.seed(675)
xgb.1 <- train(imp_team.revenue~., data=subset4, method='xgbLinear', trControl=fitCtrl, tuneGrid=tune_xgb, metric=metric)

#OOS error estimation
xgb.1.pred <- predict(xgb.1, newdata=df.test.imp)
MAE(df.test.imp$imp_team.revenue, xgb.1.pred)
RMSE(df.test.imp$imp_team.revenue, xgb.1.pred)



#====== Nearest Neighbor Search ======
drop_nn <- c(1:5,7,12:13,15)
df.nn <- df[,-drop_nn]
names(df.nn)

#impute missing values
df.nn.imp <- impute_trees(df.nn[,-1], names(df.nn)[-1])
df.nn.imp <- cbind('team.year'=df.nn[,1], df.nn.imp[,20:38], 'team.total.gms'=df[,15]) %>% data.frame()

#create derived features
df.nn.imp$city.unemploy.rate <- df.nn.imp$imp_city.unemployed / df.nn.imp$imp_city.work.force
df.nn.imp$team.avg.attend <- df.nn.imp$imp_team.total.attend / df.nn.imp$team.total.gms
df.nn.imp$team.revenue.multiple <- df.nn.imp$imp_team.value / df.nn.imp$imp_team.revenue
#df.nn.imp$city.salary.per.capita <- df.nn.imp$imp_city.salary / df.nn.imp$imp_city.returns
#transform variables
df.nn.imp$trans_team.champs.5yr <- ifelse(df.nn.imp$imp_team.champs.5yr != '0',  'Y', 'N') %>% factor()
#drop transformed variables
df.nn.imp <- subset(df.nn.imp, select=-c(imp_team.champs.5yr))

#one-hot encoding for factor variables
temp <- dummy(df.nn.imp$trans_team.champs.5yr)
df.nn.imp <- cbind(df.nn.imp, temp)
rm(temp)
df.nn.imp <- subset(df.nn.imp, select=-c(trans_team.champs.5yr))


#separate out the variables that we want to compare and the results
df.nn.x <- subset(df.nn.imp, select=-c(imp_team.revenue, imp_team.value, team.revenue.multiple))
df.nn.y <- subset(df.nn.imp, select=c(team.year, imp_team.revenue, imp_team.value, team.revenue.multiple))


#add Seattle info to the df.nn.x dataframe
arena_cap <- 18600
attend_discount <- .98
sea_values <- c(NA, 608660, NA, 5, 41*arena_cap*attend_discount, NA, NA, NA, NA, NA, NA, NA, NA, 0, NA, NA, 0, 41, 0.03, arena_cap*attend_discount, 1, 0)
#df.nn.x <- rbind(df.nn.x, sea_values) %>% data.frame()

#scale the data
df.nn.x.scaled <- scale(df.nn.x[,-1])

#find nearest neighbor
nn_output <- knn.index(df.nn.x.scaled, k=1, algorithm='kd_tree')

#find team most similar their corresponding valuation revenue multiple
comp_team <- function(index){
  comp_team_index <- nn_output[index]
  rev.multiple <- df.nn.y[c(index, comp_team_index),c(1,4)]
  return(rev.multiple)
}
comp_team(66)

team1 <- df.nn.y[seq(1,419),1] %>% data.frame()
team2 <- df.nn.y[nn_output[seq(1,419)],1] %>% data.frame()
nn.result <- cbind('team1'=team1, 'team2'=team2)
