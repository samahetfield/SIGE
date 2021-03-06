library(tidyverse)
library(funModeling)
library(ggplot2)
library(mice)
library(caret)
library(corrplot)
library(RColorBrewer)
library(NoiseFiltersR)
library(pROC)



my_roc <- function(data, predictionProb, target_var, positive_class) {
  auc <- roc(data[[target_var]], predictionProb[[positive_class]], levels = unique(data[[target_var]]))
  roc <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, main=paste('AUC:', round(auc$auc[[1]], 2)))
  return(list("auc" = auc, "roc" = roc))
}


trainRF <- function(train_data, rfCtrl = NULL, rfParametersGrid = NULL) {
  if(is.null(rfCtrl)) {
    rfCtrl <- trainControl(
      verboseIter = F, 
      classProbs = TRUE, 
      method = "repeatedcv", 
      number = 10, 
      repeats = 1, 
      summaryFunction = twoClassSummary)    
  }
  if(is.null(rfParametersGrid)) {
    rfParametersGrid <- expand.grid(
      .mtry = c(sqrt(ncol(train)))) 
  }
  
  rfModel <- train(
    target ~ ., 
    data = train_data, 
    method = "rf", 
    metric = "ROC", 
    trControl = rfCtrl, 
    tuneGrid = rfParametersGrid)
  
  return(rfModel)
}

data_raw <- read_csv('train_ok.csv')
glimpse(data_raw)
status <- df_status(data_raw)

## ---------------------------------------------------------------
## 2. Eliminar columnas no �tiles
status <- df_status(data_raw)

## columnas con NAs
na_cols <- status %>%
  filter(p_na > 70) %>%
  select(variable)
## columnas con valores diferentes
dif_cols <- status %>%
  filter(unique > 0.8 * nrow(data_raw)) %>%
  select(variable)
## eliminar columnas
remove_cols <- bind_rows(
  list(na_cols, dif_cols))
data_reduced <- data_raw %>%
  select(-one_of(remove_cols$variable))



# Visualizaci�n de la variabale Target
ggplot(data_reduced, aes(x=target)) + geom_histogram(binwidth=.5)

#Valores perdidos

# Contamos el n�mero de valores perdidos
sum(is.na(data_reduced))
colnames(data_reduced) 

data_imp <- data_reduced %>%
  na.exclude()

sum(is.na(data_imp))
sum(is.na(data_train))


# Predictor b�sico: Todos a 0
table(data_imp$target)
prop.table(table(data_imp$target))

### Realizar predicci�n con datos de test (1)
# Dado que alrededor del 90% de los valores es 0, si decimos que siempre ser� 0 la variable target, tendremos un acierto del 90%

(test <- 
    data_test %>%
    mutate(target = 0)
)
write_csv(test, "all-negative.csv")


## Filtrado de datos
filtered <-
  data_imp %>%
  filter(target == 1) %>%
  arrange(var_0)

### Histograma
# VIsualizamos la distribuci�n de los datos de var_0 cuando target == 1
ggplot(filter(data_imp, target == 1)) +
  geom_histogram(aes(x = var_0, fill = as.factor(target)), binwidth = 1)

#Correlaci�n de variables
M <-cor(data_imp)
Mone <- cor(filter(data_imp, target == 1))

corrplot(M, method = "square")
corrplot(Mone, method = "square")


# Tratamiento de outliers

#Ejemplo de c�mo se visualizan los outliers en una variable
outlier_values <- boxplot.stats(data_imp$var_199)$out
boxplot(data_imp$var_199, main="Outliers", boxwex=0.1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.6)

# Pasamos los datos a un nuevo dataframe que modificaremos
data_no_outliers <- data_imp

#Iteramos por todas las columnas para quitar los outliers
for(column in colnames(data_imp)){
  if(column != "target"){
    x <- data_imp[[column]]
    qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
    caps <- quantile(x, probs=c(.05, .95), na.rm = T)
    H <- 1.5 * IQR(x, na.rm = T)
    x[x < (qnt[1] - H)] <- caps[1]
    x[x > (qnt[2] + H)] <- caps[2]
    
    boxplot(x, main=column, boxwex=0.1)
    data_no_outliers[[column]] <- x
    
  }
  
}

# Comprobamos que realmente se han eliminado
boxplot(data_imp$var_199, main="column", boxwex=0.1)
boxplot(data_no_outliers$var_199, main="column", boxwex=0.1)

sum(is.na(data_no_outliers))

## Estudiar equilibrio de clases
table(data_imp$target)
ggplot(data_imp) + 
  geom_histogram(aes(x = target, fill = target), stat = 'count')

table(data_no_outliers$target)

## -------------------------------------------------------------------------------------
## Crear modelo de predicci�n usando Random Forest [downsampling]

predictors <- select(data_no_outliers, -target)
data_down <- downSample(x = predictors, y = data_no_outliers$target, yname = 'target')
table(data_down$target)

trainIndex <- createDataPartition(data_down$target, p = .75, list = FALSE, times = 1)
train <- data_down[ trainIndex, ] 
val   <- data_down[-trainIndex, ]
table(train$target)
table(val$target)

rfModel <- trainRF(train)
saveRDS(rfModel, file = "model2.rds")
rfModel <- readRDS("model2.rds")
print(rfModel)
prediction_p <- predict(rfModel, val, type = "prob")
prediction_r <- predict(rfModel, val, type = "raw")
result <- my_roc(val, prediction_p, "target", 1)


plotdata <- val %>%
  select(target) %>%
  bind_cols(prediction_p) %>%
  bind_cols(Prediction = prediction_r)
table(plotdata$target, plotdata$Prediction)
ggplot(plotdata) + 
  geom_bar(aes(x = target, fill = Prediction), position = position_fill())

## -------------------------------------------------------------------------------------
## Crear modelo de predicci�n usando Conditional Tree [downsampling]
library(partykit)
cTreeModel <- ctree(target ~ ., data = train)
cTreeModel
plot(cTreeModel)
predictioncT <- as.data.frame(predict(cTreeModel, newdata = val, type = "prob"))

resultcT <- my_roc(val, predictioncT, "target", 1)
