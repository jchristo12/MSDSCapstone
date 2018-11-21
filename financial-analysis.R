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

#Create helper functions
pca_var_create <- function(df, pca_obj, col_names){
  new_df <- predict(pca_obj, df)
  colnames(new_df) <- col_names
  return(new_df)
}


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
df$league.tv.deal <- as.factor(df$league.tv.deal)


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
df <- stats_df_combo(paste0(github, 'team-stats.csv'), nas, df)

#read in team stat rankings
ranking_df_combo <- function(path, na_string, orig_df){
  rankings <- read.csv(path, na.strings=nas)
  #join the data frames
  new_df <- left_join(orig_df, rankings, by='team.year')
  return(new_df)
}
df <- ranking_df_combo(paste0(github, 'team-stat-ranking.csv'), nas, df)
rank_var_names <- names(df[,57:85])


#get all the financial variables in the same units
df$city.agi <- df$city.agi/1000000
df$city.salary <- df$city.salary/1000000
df$team.salary <- df$team.salary/1000000
#put the other large numbers in terms of '000s
df$city.pop <- df$city.pop/1000
df$team.total.attend <- df$team.total.attend/1000
df$city.exempt <- df$city.exempt/1000
df$city.employed <- df$city.employed/1000
df$city.work.force <- df$city.work.force/1000
df$city.unemployed <- df$city.unemployed/1000
df$city.returns <- df$city.returns/1000


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

ggplot(df.train) +
  geom_boxplot(aes(x=league.tv.deal, y=team.value))


#team value
ggplot(df.train) +
  geom_histogram(aes(team.value), bins=30)


#number of championships in last 5yrs
ggplot(df.train) +
  geom_boxplot(aes(x=team.champs.5yr, y=team.revenue))


#correlation heatmap of team revenue, value, and team stats
df.train[,c(8,9,34:56,17)] %>%
  cor_heatmap()

#correlation heatmap of stat rankings and team.revenue
#df.train[,c(8,57:83)] %>% cor_heatmap()

#find highly correlated features
high_cor <- df.train %>%
  select_if(is.numeric) %>%
  select(-team.revenue) %>%
  cor(method='pearson', use='pair') %>%
  findCorrelation(cutoff=0.75)
df.train[,high_cor] %>% names()


#team revenue and 3pt attempted
ggplot(df.train) +
  geom_point(aes(x=X3PA, y=team.revenue, color=Age.rank))

#df.train <- df.train[,-c(30:35, 37:51, 53)]


#impute variables using decision trees
df.train.impute <- df.train %>%
  select(-c(1:3,5:7))
impute_vars <- names(df.train.impute)[-c(1, 9)]
df.train.imp <- impute_trees(df.train.impute, impute_vars)
#store the training fits to use on the test data
impute.fits <- store_impute_fit(df.train.impute, impute_vars)
rm(df.train.impute)
#combine the imputed variables to the main training dataframe
df.train.imp <- cbind(df.train[,c(1:3,5:7)], df.train.imp) %>%
  data.frame() %>%
  select(-c(impute_vars))
rank_var_names <- paste0('imp_', rank_var_names)
missing_values(df.train.imp)


#create derived features
df.train.imp$city.unemploy.rate <- df.train.imp$imp_city.unemployed / df.train.imp$imp_city.work.force
df.train.imp$team.avg.attend <- df.train.imp$imp_team.total.attend / df.train.imp$team.total.gms
df.train.imp$team.salary.per.win <- df.train.imp$imp_team.salary / df.train.imp$imp_team.wins
df.train.imp$team.revenue.multiple <- df.train.imp$imp_team.value / df.train.imp$imp_team.revenue
df.train.imp$team.salary.per.attend <- df.train.imp$imp_team.salary / df.train.imp$imp_team.total.attend
#df.train.imp$team.attend.revenue <- df.train.imp$imp_team.total.attend * df.train.imp$imp_team.ticket
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


#PCA for various data points
#stats
stats_var_names <- names(df.train.imp[,c(34:54)])
stats_df <- df.train.imp[,c(34:54)]
#fa.parallel(stats_df, fa='pc', main='Screeplot w/ Parallel Analysis', n.iter=100, show.legend=TRUE)
stats_pca_pre <- preProcess(stats_df, method=c('center', 'scale', 'pca'), pcaComp=5)
stats_pca_df <- predict(stats_pca_pre, stats_df)
#stats_pca <- principal(stats_df, nfactors=5, rotate='varimax', scores=TRUE)
#stats_pca_df <- stats_pca$scores %>% data.frame()
colnames(stats_pca_df) <- c('stats.pc1', 'stats.pc2', 'stats.pc3', 'stats.pc4', 'stats.pc5')

#tax data
tax_var_names <- names(df.train.imp[,c(22:25)])
tax_df <- df.train.imp[,c(22:25)]
#fa.parallel(tax_df, fa='pc', main='Screeplot w/ Parallel Analysis', n.iter=100, show.legend=TRUE)
tax_pca_pre <- preProcess(tax_df, method=c('center', 'scale', 'pca'), pcaComp=2)
tax_pca_df <- predict(tax_pca_pre, tax_df)
#tax_pca <- principal(tax_df, nfactors=2, rotate='varimax', scores=TRUE)
#tax_pca_df <- tax_pca$scores %>% data.frame()
colnames(tax_pca_df) <- c('tax.pc1', 'tax.pc2')

#add PCA columns to training data frame
df.train.imp <- cbind(df.train.imp, stats_pca_df, tax_pca_df) %>% data.frame()


#====== Process the Test Data ======
#impute missing data
df.test.imp <- impute_test(impute.fits, df.test, impute_vars) %>%
  select(-impute_vars)

#create derived features
df.test.imp$city.unemploy.rate <- df.test.imp$imp_city.unemployed / df.test.imp$imp_city.work.force
df.test.imp$team.avg.attend <- df.test.imp$imp_team.total.attend / df.test.imp$team.total.gms
df.test.imp$team.salary.per.win <- df.test.imp$imp_team.salary / df.test.imp$imp_team.wins
df.test.imp$team.revenue.multiple <- df.test.imp$imp_team.value / df.test.imp$imp_team.revenue
df.test.imp$team.salary.per.attend <- df.test.imp$imp_team.salary / df.test.imp$imp_team.total.attend
#df.test.imp$team.attend.revenue <- df.test.imp$team.total.attend * df.test.imp$imp_team.ticket
#df.test.imp$city.salary.per.capita <- df.test.imp$imp_city.salary / df.test.imp$imp_city.returns

#transform variables
df.test.imp$trans_team.champs.5yr <- ifelse(df.test.imp$imp_team.champs.5yr != '0',  'Y', 'N') %>% factor()
#df.test.imp$trans_city.returns <- log(df.test.imp$imp_city.returns)
#df.test.imp$trans_city.exempt <- log(df.test.imp$imp_city.exempt)

#add PCA data
#stats data
stats_test <- predict(stats_pca_pre, newdata=df.test.imp)[,71:75]
colnames(stats_test) <- c('stats.pc1', 'stats.pc2', 'stats.pc3', 'stats.pc4', 'stats.pc5')
#tax data
tax_test <- predict(tax_pca_pre, newdata=df.test.imp)[,88:89]
colnames(tax_test) <- c('tax.pc1', 'tax.pc2')
#combine data
df.test.imp <- cbind(df.test.imp, stats_test, tax_test) %>% data.frame()


#====== Data Modeling ======
#set the training parameters
fitCtrl <- trainControl(method="repeatedcv", number=10, repeats=3)
rfe_control <- rfeControl(functions=rfFuncs, method='cv', number=10)
metric <- 'RMSE'

#universal unneeded variables for modeling
droplist <- c('city', 'team', 'nickname', 'city.year', 'team.year', 'nickname.year', 'imp_team.value', 'team.total.gms', 'team.revenue.multiple', 'year',
              rank_var_names, stats_var_names, tax_var_names)
orig_var <- c('imp_team.champs.5yr')
other_drop_vars <- c('team.attend.revenue', 'team.superstar.cat', 'imp_G', 'imp_PTS.rank', 'imp_MP')
high_cor_vars <- c('imp_team.fci', 'imp_city.salary', 'imp_city.agi', 'imp_city.exempt', 'imp_team.total.attend', 'imp_city.pop', 'imp_city.employed',
                   'imp_city.work.force', 'team.salary.per.win', 'team.salary.per.attend', 'imp_team.superstar')

#Linear Regression
#drop unneeded features
subset1 <- create_subset(df.train.imp, c(droplist, orig_var, other_drop_vars, high_cor_vars, 'imp_pts.per.gm', 'imp_salary.per.pt'))
subset1 %>%
  select_if(is.numeric) %>%
  cor_heatmap()

#recursive feature elimination
set.seed(45)
rfe_results <- subset1 %>%
  select(-imp_team.revenue) %>%
  rfe(y=log(subset1$imp_team.revenue), sizes=c(1:dim(subset1)[2]), rfeControl=rfe_control, metric=metric)
#list the top predictors
predictors(rfe_results)

#build the linear regression model
model_data1 <- subset1 %>%
  select(predictors(rfe_results)) %>%
  cbind('imp_team.revenue'=subset1$imp_team.revenue) %>%
  data.frame()
#model_data1$imp_league.tv.deal <- subset1$imp_league.tv.deal
model_data1$stats.pc1 <- subset1$stats.pc1
#model_data1$stats.pc3 <- subset1$stats.pc3

lin.mod.1 <- lm(log(imp_team.revenue)~.-imp_city.unemployed-imp_team.ticket-tax.pc2-imp_city.franchises, data=model_data1)

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

#features used in model
features <- names(lin.mod.1$coefficients)[-1]



#====== TESTING ======
#seattle_data <- load_seattle_data()
#tax_sea <- pca_var_create(seattle_data[,3:6], tax_pca_pre, c('tax.pc1', 'tax.pc2'))
#stats_sea <- pca_var_create(seattle_data[,15:35], stats_pca_pre, c('stats.pc1', 'stats.pc2', 'stats.pc3', 'stats.pc4', 'stats.pc5'))
#seattle_data_full <- cbind(seattle_data, tax_sea, stats_sea) %>% data.frame()

#test_model <- lm(log(imp_team.revenue)~., data=subset(model_data1, select=c('imp_team.revenue', features)))
#summary(test_model)
#predict(test_model, newdata=seattle_data_full) %>% exp()



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
drop.sub3 <- c('year', 'imp_team.fci', 'imp_city.salary', 'trans_city.exempt', 'trans_city.returns', 'imp_team.total.attend')
subset3 <- create_subset(df.train.imp, c(droplist, orig_var, other_drop_vars, drop.sub3))
names(subset3)
var_floor <- sqrt(ncol(subset3)-1) %>% floor()
grid <- expand.grid(.mtry=1:20)
rf_cv <- train(imp_team.revenue~., data=subset3, method='rf', metric=metric, trControl=fitCtrl, tuneGrid=grid, importance=TRUE)

rf <- randomForest(imp_team.revenue~., data=subset3, ntree=1000, mtry=10, importance=TRUE)
varImp(rf_cv)
varImpPlot(rf, main="Variable Importance Plot")

#OOS (rf) error estimation
rf.pred <- predict(rf, newdata=df.test.imp)
MAE(df.test.imp$imp_team.revenue, rf.pred)
RMSE(df.test.imp$imp_team.revenue, rf.pred)

#OOS (rf_cv) error estimation
rf_cv_pred <- predict(rf_cv, newdata=df.test.imp)
MAE(df.test.imp$imp_team.revenue, rf_cv_pred)
RMSE(df.test.imp$imp_team.revenue, rf_cv_pred)


#XGBoost
subset4 <- create_subset(df.train.imp, c(droplist, orig_var, other_drop_vars))
#perform one hot encoding
#subset4 <- one_hot_encode(subset4)


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



#====== Final Franchise Revenue Model ======
#impute variables using decision trees
df_final_impute <- df %>%
  select(-c(1:3,5:7))
impute_vars_final <- names(df_final_impute)[-c(1, 9)]
df_final_imp <- impute_trees(df_final_impute, impute_vars_final)
rm(df_final_impute)
#combine the imputed variables to the main training dataframe
df_final_imp <- cbind(df[,c(1:3,5:7)], df_final_imp) %>%
  data.frame() %>%
  select(-c(impute_vars))

#create derived variables
df_final_imp$team.avg.attend <- df_final_imp$imp_team.total.attend / df_final_imp$team.total.gms

#create PCA variables
#PCA for various data points
#stats
stats_final_df <- df_final_imp[,c(34:54)]
stats_pca_obj <- preProcess(stats_final_df, method=c('center', 'scale', 'pca'), pcaComp=5)
stats_pca_final_df <- predict(stats_pca_obj, stats_final_df)
colnames(stats_pca_final_df) <- c('stats.pc1', 'stats.pc2', 'stats.pc3', 'stats.pc4', 'stats.pc5')
#tax data
tax_final_df <- df_final_imp[,c(22:25)]
tax_pca_obj <- preProcess(tax_final_df, method=c('center', 'scale', 'pca'), pcaComp=2)
tax_pca_final_df <- predict(tax_pca_obj, tax_final_df)
colnames(tax_pca_final_df) <- c('tax.pc1', 'tax.pc2')
#add PCA columns to final data frame
df_final_imp <- cbind(df_final_imp, stats_pca_final_df, tax_pca_final_df) %>% data.frame()

#Calculate final model parameters
final_model_df <- subset(df_final_imp, select=features)
final_model_df$imp_team.revenue <- df_final_imp$imp_team.revenue

final_model <- lm(log(imp_team.revenue)~., data=final_model_df)

#create function to ingest seattle data and prep to run through model
rev_func <- function(seattle_data){
  #prep the tax data
  tax <- seattle_data[,3:6]
  tax_df <- predict(tax_pca_obj, tax)
  colnames(tax_df) <- c('tax.pc1', 'tax.pc2')
  #prep the stats data
  stats <- seattle_data[,19:39]
  stats_df <- predict(stats_pca_obj, stats)
  colnames(stats_df) <- c('stats.pc1', 'stats.pc2', 'stats.pc3', 'stats.pc4', 'stats.pc5')
  #add pca variables to data for modeling
  model_data <- cbind(seattle_data, tax_df, stats_df) %>% data.frame()
  #make the predictions
  output <- predict(final_model, newdata=model_data, interval='predict') %>% exp()
  return(output)
}


#====== Final Valuation Model ======
drop_final_nn <- c(1:10,13:14,16,32:33,55:93)
stats_nn <- c(34:54)
final_nn_df <- df_final_imp[,-c(drop_final_nn, stats_nn)]
final_nn_df <- cbind(final_nn_df, 'team.avg.attend'=df_final_imp$team.avg.attend) %>% data.frame()
final_nn_df$city.umemploy.rate <- final_nn_df$imp_city.unemployed / final_nn_df$imp_city.work.force
final_nn_df$trans_team.champs.5yr <- ifelse(final_nn_df$imp_team.champs.5yr != '0',  'Y', 'N') %>% factor()
final_nn_df <- create_subset(final_nn_df, c('imp_team.champs.5yr', 'imp_city.work.force', 'imp_city.employed', 'imp_city.unemployed', 'imp_team.ticket',
                                            'imp_team.fci'))
#one hot encoding
te <- dummy(final_nn_df$trans_team.champs.5yr)
final_nn_df <- cbind(final_nn_df, te) %>% data.frame()
rm(te)
final_nn_df <- subset(final_nn_df, select=-c(trans_team.champs.5yr))

te <- dummy(final_nn_df$imp_league.tv.deal)
final_nn_df <- cbind(final_nn_df, te) %>% data.frame()
rm(te)
final_nn_df <- subset(final_nn_df, select=-c(imp_league.tv.deal))

#create the revenue, value, and multiple data frame
df_final_imp$rev.multiple <- df_final_imp$imp_team.value / df_final_imp$imp_team.revenue
nn_key <- subset(df_final_imp, select=c(team.year, rev.multiple, imp_team.revenue, imp_team.value))

#reorder dataframe columns
#final_nn_df <- cbind(final_nn_df[,-c(2,4,11:32)], final_nn_df[,c(2,4,11:32)]) %>% data.frame()
final_nn_df <- cbind(final_nn_df[,-c(2,4,11)], final_nn_df[,c(2,4,11)]) %>% data.frame()

value_func <- function(seattle_data){
  seattle_data <- seattle_data[,-c(19:39)]
  others <- dim(final_nn_df)[1]
  start <- others + 1
  sea_length <- dim(seattle_data)[1]
  #initialize a data frame
  full_mult <- matrix(ncol=1, nrow=0) %>%
    data.frame()
  full_comp <- matrix(ncol=1, nrow=0) %>%
    data.frame()
  #add seasons one by one to get their match
  i <- 1
  k <- 3
  while(i <= sea_length){
    all_df <- rbind(final_nn_df, seattle_data[i,]) %>% data.frame()
    #scale the data
    nn_scaled <- scale(all_df, center=TRUE, scale=TRUE)
    #create the final KNN model
    final_nn_output <- knn.index(nn_scaled, k=k, algorithm='kd_tree')
    comp_team <- final_nn_output[start,c(1:k)]
    #find the revenue multiple for the seattle teams
    result <- nn_key[comp_team,]
    mean_mult <- mean(result[,2])
    top_comp <- nn_key[comp_team,1][1]
    output <- c(top_comp, mean_mult)
    full_mult <- rbind(full_mult, mean_mult) %>% data.frame()
    full_comp <- rbind(full_comp, top_comp) %>% data.frame()
    i <- i + 1
  }
  full_return <- cbind(full_comp, full_mult) %>% data.frame()
  colnames(full_return) <- c('compTeam', 'avgMultiple')
  return(full_return)
}


#====== Testing ======
load_seattle_data <- function(){
  df <- read.csv('https://raw.githubusercontent.com/jchristo12/MSDSCapstone/master/seattle_data.csv')
  return(df)
}
seattle_data <- load_seattle_data()
rev_func(seattle_data)
value_func(seattle_data)

#objects to keep in workspace
keep <- c('rev_func', 'value_func', 'load_seattle_data', 'final_model', 'tax_pca_obj', 'stats_pca_obj', 'final_nn_df', 'nn_key')

#REMOVES ALL OBJECTS EXCEPT WHAT IS SPECIFIED ABOVE
rm(list=ls()[!(ls() %in% keep)])
