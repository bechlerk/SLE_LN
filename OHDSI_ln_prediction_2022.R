devtools::install_github("jdposada/BQJdbcConnectionStringR", upgrade = "never")

bqDriverPath = "/home/ohdsi/workdir/BQ_driver"
work_project_id = 'som-nero-nigam-starr'
work_dataset_id = "kbechler_explore"
jsonPath = "/home/ohdsi/workdir/application_default_credentials.json"
cdm_project_id = "som-nero-nigam-starr"
cdm_dataset_id = "starr_omop_cdm5_deid_20211213"
cdmDatabaseSchema <- "`som-nero-nigam-starr.starr_omop_cdm5_deid_20211213`"
cohortDatabaseSchema <- work_dataset_id
resultsDatabaseSchema = 'som-nero-nigam-starr.kbechler_explore'

# Database description
databaseId <- "STARROMOP"
databaseName <- "STAnford medicine Research data Repository (STARR-OMOP)"


#checkPlpInstallation(connectionDetails = NULL, python = T)

#Sys.which("conda")
#reticulate::install_miniconda()
#configurePython(envname='r-reticulate', envtype='conda')


connectionString <-  BQJdbcConnectionStringR::createBQConnectionString(projectId = work_project_id,
                                                                       defaultDataset = work_dataset_id,
                                                                       authType = 2,
                                                                       jsonCredentialsPath = jsonPath)
# Test BQ Connection
connectionDetails <- DatabaseConnector::createConnectionDetails(dbms="bigquery",
                                                                connectionString=connectionString,
                                                                user="",
                                                                password='',
                                                                pathToDriver = bqDriverPath)

connection <- DatabaseConnector::connect(connectionDetails)

# Install package and load libraries
library(drat)
drat::addRepo("OHDSI")
library(FeatureExtraction)
library(dplyr)
library(PatientLevelPrediction)
library(Andromeda)
library(ggplot2)
library(htmltools)
library(reticulate)

# Use default set of covariates
#settings <- createDefaultTemporalCovariateSettings()
settings = createDefaultCovariateSettings()


covariateData <- getDbCovariateData(connectionDetails = connectionDetails,
                                    cdmDatabaseSchema = cdmDatabaseSchema,
                                    cohortDatabaseSchema = 'som-nero-nigam-starr.kbechler_explore',
                                    cohortTable = "exp_train_14",
                                    cohortId = 21,
                                    rowIdField = "subject_id",
                                    covariateSettings = settings, 
                                    aggregated = FALSE)

summary(covariateData)

# Per person covariate output format
covariateData$covariates

# Save covariate data
saveCovariateData(covariateData, "workdir/covariates")
covariateData <- loadCovariateData("workdir/covariates")
dim(covariateData)


# Remove infrequent covariates: remove features that have non-zero values for only one or a few subjects in cohort
# Increase computational burden, but unlikely to end up in any fitted model
# By default, covariates appearing in less than 0.1% of the subjects are removed
# Normalization: Scales all covariate values to a value between 0 and 1 (by dividing by the max value for each covariate)
# Removal of redundancy: Remove covariates if every person in the cohort has the same value for a covariate
# Pose problem for some ML algorithms, causing simple regression to no longer have a single solution

tidyCovariates <- tidyCovariateData(covariateData,
                                    minFraction = 0.001,
                                    normalize = TRUE,
                                    removeRedundancy = TRUE)
summary(tidyCovariates)
summary(covariateData)

# How many infrequent covariates were removed
deletedCovariateIds <- attr(tidyCovariates, "metaData")$deletedInfrequentCovariateIds

# Which redundant covariates were removed
deletedCovariateIds <- attr(tidyCovariates, "metaData")$deletedRedundantCovariateIds

library(Andromeda)
covariateData$covariateRef %>%
  filter(covariateId %in% deletedCovariateIds, ) %>%
  collect()

summary(covariateData)


# Create aggregated covariates for cohort of interest
#covariateData2 = aggregateCovariates(covariateData)

# Create table of overall study population characteristics
#result = createTable1(covariateData2)
#print(result, row.names = FALSE, right = FALSE)

plpData <- getPlpData(connectionDetails = connectionDetails,
                      cdmDatabaseSchema = cdmDatabaseSchema,
                      cohortDatabaseSchema = resultsDatabaseSchema,
                      oracleTempSchema = NULL,
                      cohortTable = 'exp_train_14',
                      cohortId = 21,
                      covariateSettings = settings,
                      outcomeDatabaseSchema = resultsDatabaseSchema,
                      outcomeTable = 'out_train_14',
                      outcomeIds = 22, 
                      firstExposureOnly = TRUE)

savePlpData(plpData, "workdir/sle_ln_data_train")
plpData = loadPlpData("workdir/sle_ln_data_train")
summary(plpData)
plpData$cohorts
plpData$outcomes
plpData$covariataData
plpData$timeRef


# Create study population function from OHDSI
population = createStudyPopulation(plpData = plpData,
                                   outcomeId = 22,
                                   binary = TRUE, #force to be binary prediction problem
                                   includeAllOutcomes = TRUE,
                                   firstExposureOnly = TRUE)
                                   #riskWindowStart = 1, 
                                   #startAnchor = "cohort start")
                                   #washoutPeriod
                                   #removeSubjectsWithPriorOutcome = TRUE, #should already be done
                                   #priorOutcomeLookback,
                                   #requireTimeAtRisk,
                                   #minTimeAtRisk, #should already be done
                                   #riskWindowStart = 1 #start of risk window (in days) relative to the startAnchor
                                   #startAnchor = "cohort start")
                                   #riskWindowEnd = 99999,
                                   #endAnchor = "cohort end", 
                                   #verbosity = DEBUG, TRACE, INFO, WARN, ERROR, FATAL
                                   #restrictTarToCohortEnd = T)

summary(population)

# Logisitc Regression model
lrmodel = setLassoLogisticRegression(variance = 0.01, seed = 1234)
lrResults = runPlp(population, plpData, minCovariateFraction = 0.1, normalizeData = TRUE, 
                   modelSettings = lrmodel, testSplit = 'stratified', 
                   testFraction = 0.25, nfold = 5, splitSeed = 1234)
logreg_model = lrResults$model
savePlpModel(lrResults$model, "workdir/logreg_model")
logreg_model = loadPlpModel("workdir/logreg_model")
savePlpResult(lrResults, "logreg_results")
lrResults = loadPlpResult("logreg_results")
summary(lrResults)
lrResults$executionSummary

# Random forest model

rf_model = setRandomForest()
rfResults = runPlp(population, plpData, modelSettings = rf_model, testSplit = 'stratified', 
                   testFraction = 0.25, nfold = 2, splitSeed = 1234)
rf_model = rfResults$model
savePlpModel(rfResults$model, "workdir/rf_model")
rf_model = loadPlpModel("workdir/rf_model")
savePlpResult(rfResults, "rf_results")
rfResults = loadPlpResult("rf_results")


# Gradient Boosting Model
gbm_model = setGradientBoostingMachine(ntrees = c(100, 1000), earlyStopRound = 25, 
                                       maxDepth = c(1, 4, 6, 10, 15, 18), minRows = 2, 
                                       learnRate = c(0.005, 0.01, 0.1, 0.5), seed = 1234)
gbmResults = runPlp(population, plpData, modelSettings = gbm_model, 
                    minCovariateFraction = 0.1, testSplit = 'stratified', 
                    testFraction = 0.25, nfold = 5, splitSeed = 1234)
gbm_model = gbmResults$model
savePlpModel(gbmResults$model, "workdir/gbm_model")
gbm_model = loadPlpModel("workdir/gbm_model")
savePlpResult(gbmResults, "gbm_results")
gbmResults = loadPlpResult("gbm_results")


# Run on held out test data
covariateData_test <- getDbCovariateData(connectionDetails = connectionDetails,
                                         cdmDatabaseSchema = cdmDatabaseSchema,
                                         cohortDatabaseSchema = 'som-nero-nigam-starr.kbechler_explore',
                                         cohortTable = "exp_test_14",
                                         cohortId = 21,
                                         rowIdField = "subject_id",
                                         covariateSettings = settings, 
                                         aggregated = FALSE)

saveCovariateData(covariateData_test, "workdir/covariates_test")
covariateData_test = loadCovariateData("workdir/covariates_test")
summary(covariateData_test)

tidyCovariates_test <- tidyCovariateData(covariateData_test,
                                    minFraction = 0.001,
                                    normalize = TRUE,
                                    removeRedundancy = TRUE)


# How many infrequent covariates were removed
deletedCovariateIds_test <- attr(tidyCovariates_test, "metaData")$deletedInfrequentCovariateIds

# Which redundant covariates were removed
deletedCovariateIds_test <- attr(tidyCovariates_test, "metaData")$deletedRedundantCovariateIds

plpData_test <- getPlpData(connectionDetails = connectionDetails,
                           cdmDatabaseSchema = cdmDatabaseSchema,
                           cohortDatabaseSchema = resultsDatabaseSchema,
                           oracleTempSchema = NULL,
                           cohortTable = 'exp_test_14',
                           cohortId = 21,
                           covariateSettings = settings,
                           outcomeDatabaseSchema = resultsDatabaseSchema,
                           outcomeTable = 'out_test_14',
                           outcomeIds = 22, 
                           firstExposureOnly = TRUE)



savePlpData(plpData_test, "workdir/sle_ln_data_test")
plpData_test = loadPlpData("workdir/sle_ln_data_test")



population_test = createStudyPopulation(plpData = plpData_test,
                                        outcomeId = 22,
                                        binary = TRUE,
                                        includeAllOutcomes = TRUE,
                                        firstExposureOnly = TRUE)

# Apply model
lr_prediction = applyModel(population = population_test,
                           plpData = plpData_test,
                           plpModel = logreg_model,
                           calculatePerformance = T)

plotPrecisionRecall(lr_prediction)



viewPlp(lr_prediction)


savePlp
reticulate::py_config()

gbm_prediction = applyModel(population = population_test, 
                           plpData = plpData_test, 
                           plpModel = gbm_model, 
                           calculatePerformance = T)

gbm_prediction$model$modelSettings
lr_prediction$model$modelSettings

# Export predictions
gbm_output = gbm_prediction$prediction
lr_output = lr_prediction$prediction

write.csv(gbm_output,"gbm_output.csv", row.names = FALSE)
write.csv(lr_output,"gbm_output.csv", row.names = FALSE)

dim(gbm_output)

gb_pred = gbm_output$outcomeCount





#create confusion matrix and calculate metrics related to confusion matrix
confusionMatrix(pred, actual, mode = "everything", positive="1")

