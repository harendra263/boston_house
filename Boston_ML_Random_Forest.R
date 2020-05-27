
# Steps in Applied Machine learning and Data Science
# 1. Load library
# 2. Load Datasets to which machine learning algorithm to be applied
    # Either a) Load from CSV file or b) load from Database
# 3. Summarization of datasets to understand datasets (Descriptive)
# 4. Visualization of Datasets to understand datasets (Plots, Graphs)
# 5. Data Preprocessing and Data transformation (Split into train and test datasets)
# 6. Application of Machine learning algorithm to training datasets
###  a)setup a ML Algorithm and paramter settings
###  b)cross validation setup with training datasets
###  c) training and fitting Algorithm with training datasets
###  d) evaluation of trained Algorithm (or Model) and result
###  e) saving the trained model for future prediciton
# 7. Load the saved Model and apply it to new datasets for future prediction

# load Necessary Libraries

library(DBI)
library(corrgram)
library(caret)
library(gridExtra)
library(ggpubr)
library(reshape2)

# Turn of the warning

options(warn = -1)

# Load the Datasets : housing training data

setwd('D:\\Kaggle Competition\\boston_housing_datasets')

datasets <- read.csv('Boston.csv', header = TRUE,sep = ',')[-1]

colnames(datasets)

colnames(datasets) <- c('CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD',
                        'TAX','PTRATIO','BLACK','LSAT','MVAL')

colnames(datasets)

# Print top 10 rows in the datasets

head(datasets,10)

# Print bottom 10 rows in the datasets

tail(datasets,10)

# Dimension of the datasets

dim(datasets)

# Check the datatypes for each column

table(unlist(lapply(datasets, class)))


# Check the datatypes of each  individual features

data.class(datasets$CRIM)
data.class(datasets$ZN)
data.class(datasets$INDUS)
data.class(datasets$CHAS)
data.class(datasets$NOX)
data.class(datasets$RM)
data.class(datasets$AGE)
data.class(datasets$DIS)
data.class(datasets$RAD)
data.class(datasets$PTRATIO)
data.class(datasets$BLACK)
data.class(datasets$LSAT)
# Target
data.class(datasets$MVAL)


# Check the datatypes for each column

table(unlist(lapply(datasets, class)))

# Exploring or Summarising the datasets with descriptive statistics

# Find out if there missing value

colSums(is.na(datasets))

# Checking missing values using different packages

library(mice)
summary(datasets)
md.pattern(datasets,plot = TRUE)

# Using VIM packages
# install.packages('VIM')
library(VIM)
mice_plot <- aggr(datasets,col = c('navyblue','yellow'),
                  numbers = TRUE,sortVars = TRUE, labels = names(datasets[1:14]),
                  cex.axis = 0.9,gap = 3,ylab = c('Missing Data','Pattern'))

# ---------------------------------
# Summary of datasets
# ---------------------------------

numericalCols <- c(1:14)

# lapply- When you want to apply a function to each element of a list  in turn and get a 
# list back

lapply(datasets[numericalCols], FUN = sum)
lapply(datasets[numericalCols], FUN = mean)
lapply(datasets[numericalCols], FUN = median)
lapply(datasets[numericalCols], FUN = min)
lapply(datasets[numericalCols], FUN = max)
lapply(datasets[numericalCols], FUN = length)

# sapply =  When you want to apply  a function to each element of a list in turn ,
# you want a  vector back rather than a list

sapply(datasets[numericalCols], FUN = sum)
sapply(datasets[numericalCols], FUN = mean)
sapply(datasets[numericalCols], FUN = median)
sapply(datasets[numericalCols], FUN = min)
sapply(datasets[numericalCols], FUN = max)
sapply(datasets[numericalCols], FUN = length)


# Correlation and Covariances among numerical variables
library(Hmisc)
cor(datasets[numericalCols],use = "complete.obs",method = "kendall")
cov(datasets[numericalCols], use = 'complete.obs')


#  Correlation with significance levels

rcorr(as.matrix(datasets[numericalCols]),type = "pearson")

# Scatterplot matrices from glus package

library(gclus)
dta <- datasets[1:14]
dta.r <- abs(cor(dta))  # get correlations
dta.col <- dmat.color(dta.r) # get colors

# reorder variables so those with highest corelation are closest to the diagonal
dta.o <- order.single(dta.r)
cpairs(dta,dta.o,panel.colors = dta.col,gap = 0.5,main = "Variables Ordered and Colored by 
       Correlation")


# -------------------------------------------
# Visualizing datasets
# ------------------------------------------

# Histogram using ggplot

ggplot(data = melt(datasets[1:14]),mapping = aes(x = value))  +
  geom_histogram(bins = 30) + facet_wrap(~variable,scales = 'free_x')


# More graphs on correlation data
# using "Hmisc


res2 <- rcorr(as.matrix(datasets[,c(1:14)]))
print(res2)
# Extract the correlation coefficients
res2$r
# Extract p-values
res2$P

# Using corrplot
library(corrplot)
library(RColorBrewer)

corrplot(res2$r,type = "upper",order = "hclust",col = brewer.pal(n = 8,name = "RdYlBu"),
         tl.col = "black",tl.srt = 45)


corrplot(res2$r,type = "lower",order = "hclust",col = brewer.pal(n = 8,name = "RdYlBu"),
         tl.col = "black",tl.srt = 45)

# Using cor_function

M <- cor(datasets[,c(1:14)])
corrplot(M,type = "upper",order = "hclust",col = brewer.pal(n = 8,name = "RdYlBu"))

# Insignicant Correlation are crossed

corrplot(res2$r,type = "upper",order = "hclust",p.mat = res2$P,sig.level = 0.01,insig = "pch")
corrplot(res2$r,type = "lower",order = "hclust",p.mat = res2$P,sig.level = 0.01,insig = "pch")

# Insignicant Correlation are left blanked

corrplot(res2$r,type = "upper",order = "hclust",p.mat = res2$P,sig.level = 0.01,insig = "blank")
corrplot(res2$r,type = "lower",order = "hclust",p.mat = res2$P,sig.level = 0.01,insig = "blank")


# Using Colored Heatmap

col <- colorRampPalette(c("blue","white","red"))(20)
heatmap(x = res2$r,col=col,symm = TRUE)

# Scatterplot matrices from glus package

library(gclus)
dta <- datasets[,c(3,5:8,10,12)]
dta.r <- abs(cor(dta))  # get correlations
dta.col <- dmat.color(dta.r) # get colors

# reorder variables so those with highest corelation are closest to the diagonal
dta.o <- order.single(dta.r)
cpairs(dta,dta.o,panel.colors = dta.col,gap = 0.5,main = "Variables Ordered and Colored by 
       Correlation")

# ----------------------------
# Visualise correlation
# ----------------------------
library(corrgram)
corrgram(datasets[c(3,5:8,10,12)],order = TRUE,lower.panel = panel.shade,
         upper.panel = panel.pie, text.panel = panel.txt,main  = " ")


# More graphs on correlation data
# using "Hmisc


res2 <- rcorr(as.matrix(datasets[,c(3,5:8,10,12)]))
print(res2)
# Extract the correlation coefficients
res2$r
# Extract p-values
res2$P

# ========================================================================
#  Preprocessing of datasets i.e Train-Test Split
# ========================================================================

train_test_index <- createDataPartition(datasets$MVAL,p = 0.75,list = FALSE)
training_dataset <- datasets[,c(1:14)][train_test_index,]
testing_dataset <- datasets[,c(1:14)][-train_test_index,]

dim(training_dataset)
dim(testing_dataset)

# =============================================
# cross-validation and control parameter setup
# =============================================
#repeatedcv/adaptivecv

control <- trainControl(method = 'repeatedcv',
                        number = 3,repeats = 3,
                        verbose = TRUE,search = 'grid',
                        allowParallel = TRUE)
metric <- "Accuracy"
tuneLength = 10

# ==========================================================
# Machine learning and parameter tuning
# 1. Without parameter  tuning or  using default
#  
# There are three ways of parameter tuning
# 2. Using Data Preprocessing :
# caret Method <- preprocess
# default value is null
# other value  ["BoxCox","YeoJohnson","expoTrans","centre","scale","range","knnimpute",
#              "bagimpute","medianimpute","pca","ica" ans "spatialSign"]

#  Using Automatic Grid
#  caret Method =tuneLength[Note :- it takes an integer value]
#  Example tuneLength==4
# 
# Using Manual Grid
#  caret method <-  tuneLength [Note:grid needs to be defined]
#  Example : grid <- expand.grid(size =c(5,10),k=c(3,4,5)) [parameters of LVQ]
#               tuneLength = grid
# ==============================================================
# =============================================================


# -----------------------------------------------------------------
# Training without explicit parameter tuning / using default
# -----------------------------------------------------------------
#  01. Random Forest : cForest-Regression (Supervised Learning)
fit.cForest_1 <- caret::train(MVAL~., data = training_dataset,method = "cforest",
                            metric = "RMSE", trControl=control)

print(fit.cForest_1)
# plot(fit.cForest_1)
# importance <- varImp(fit.cForest_1, scale=FALSE)
# print(importance);plot(importance)

# 0.2 Random Forest : parRF - Regression (Supervised Learning)

fit.parRF_1 <- caret::train(MVAL~.,data = training_dataset,method = "parRF",
                             metric = "RMSE", trControl=control)

print(fit.parRF_1)
plot(fit.parRF_1)

# 03.Bagged CART : treebag - Regression (Supervised Learning)
fit.treebag_1 <- caret::train(MVAL ~ . , data = training_dataset,method = "treebag",
                              metric = "RMSE",trControl = control)
print(fit.treebag_1)

# 04. Quantile Random Forest : qrf - Regression (Supervised Learning)

fit.qrf_1 <- caret::train(MVAL ~ . , data= training_dataset,method="qrf",
                          metric= "RMSE",trControl=control)
print(fit.qrf_1)
# plot(fit.qrf_1)

# ================================================================
# Training with explicit paramter tuning using preProcess Method
# ================================================================

#  01. Random Forest : cForest-Regression (Supervised Learning)
fit.cForest_2 <- caret::train(MVAL~., data = training_dataset,method = "cforest",
                              metric = "RMSE", trControl=control,
                              preProcess=c("center","scale"))

print(fit.cForest_2)
# plot(fit.cForest_2)
# importance <- varImp(fit.cForest_2, scale=FALSE)
# print(importance);plot(importance)

# 0.2 Random Forest : parRF - Regression (Supervised Learning)

fit.parRF_2 <- caret::train(MVAL~., data = training_dataset,method = "parRF",
                            metric = "RMSE", trControl=control,
                            preProcess = c("center","scale"))

print(fit.parRF_2)
# plot(fit.parRF_2)

# 03.Bagged CART : treebag - Regression (Supervised Learning)
fit.treebag_2 <- caret::train(MVAL ~ . , data= training_dataset,method="treebag",
                              metric= "RMSE",trControl=control,
                              preProcess = c("center","scale"))
print(fit.treebag_2)

# 04. Quantile Random Forest : qrf - Regression (Supervised Learning)

fit.qrf_2 <- caret::train(MVAL ~ . , data= training_dataset,method="qrf",
                          metric= "RMSE",trControl=control,
                          preProcess = c("center","scale"))
print(fit.qrf_2)
# plot(fit.qrf_2)

# ================================================================
# Training with explicit paramter tuning using preProcess Method
# ================================================================

#  01. Random Forest : cForest-Regression (Supervised Learning)
fit.cForest_3 <- caret::train(MVAL~., data = training_dataset,method = "cforest",
                              metric = "RMSE", trControl=control,
                              preProcess=c("center","scale"),tuneLength=3)

print(fit.cForest_3)
# plot(fit.cForest_3)
# importance <- varImp(fit.cForest_3, scale=FALSE)
# print(importance);plot(importance)

# 0.2 Random Forest : parRF - Regression (Supervised Learning)

fit.parRF_3 <- caret::train(MVAL~., data = training_dataset,method = "parRF",
                            metric = "RMSE", trControl=control,
                            preProcess = c("center","scale"),tuneLength=3)

print(fit.parRF_3)
# plot(fit.parRF_3)

# 03.Bagged CART : treebag - Regression (Supervised Learning)
fit.treebag_3 <- caret::train(MVAL ~ . , data= training_dataset,method = "treebag",
                              metric = "RMSE",trControl = control,
                              preProcess = c("center","scale"),tuneLength = 3)
print(fit.treebag_3)

# 04. Quantile Random Forest : qrf - Regression (Supervised Learning)

fit.qrf_3 <- caret::train(MVAL ~ . , data= training_dataset,method="qrf",
                          metric= "RMSE",trControl=control,
                          preProcess = c("center","scale"),tuneLength = 3)
print(fit.qrf_3)
# plot(fit.qrf_3)


# ===========================================================
# collect the results of trained model
# ===========================================================

results <- resamples(list(CF_1 = fit.cForest_1,
                          parRF_1 = fit.parRF_1,
                          TreeBag_1 = fit.treebag_1,
                          QRF_1 = fit.qrf_1,
                          
                          CF_2 = fit.cForest_2,
                          parRF_2 = fit.parRF_2,
                          TreeBag_2 = fit.treebag_2,
                          QRF_2 = fit.qrf_2,
                          
                          CF_3 = fit.cForest_3,
                          parRF_3 = fit.parRF_3,
                          TreeBag_3 = fit.treebag_3,
                          QRF_3 = fit.qrf_3
))

summary(results)

# plot and rank the fitted models

dotplot(results)
bwplot(results)

# assign the best trained model
best_trained_model <- fit.parRF_3

# ========================================================
# Save the model to the disk
# ========================================================

getwd()
saveRDS(best_trained_model,"./best_trained_model.rds")

# load the model

getwd()
saved_model <- readRDS(("./best_trained_model.rds"))
print(saved_model)

# make  predictions on the new datasets using final model

final_predictions <- predict(saved_model,testing_dataset[,c(1:13)])

install.packages("Metrics")
library(Metrics)
mse(testing_dataset[,c(14)], final_predictions)
rmse(testing_dataset[,c(14)], final_predictions)











