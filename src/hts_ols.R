library(hts)

reconcile_all <- function(
        data_dir = 'data', 
        in_dir = 'preds', 
        out_dir = 'reconciled_preds', 
        min_trace = TRUE,
        horizon = 12
){
        
        if (!dir.exists(file.path(data_dir, out_dir))) dir.create(file.path(data_dir, out_dir))
        tourism_df <- read.csv(file.path(data_dir, 'raw/TourismData_v3.csv'))
        tourism_gts <- gts(tourism_df[,3:ncol(tourism_df)], characters = list(c(1,1,1), c(3)))
        
        for (filename in list.files(file.path(data_dir, in_dir))) {
                forecasts <- read.csv(file.path(data_dir, in_dir, filename))
                if (min_trace) {
                        reconciled_forecasts <- MinT(
                                forecasts[(nrow(forecasts) - horizon + 1):nrow(forecasts),], 
                                groups = tourism_gts$groups, 
                                residual = as.matrix(forecasts[1:(nrow(forecasts) - horizon),]),
                                covariance = 'shr',
                                keep = 'all'
                        )
                }
                else {
                        reconciled_forecasts <- combinef(
                                forecasts[(nrow(forecasts) - horizon + 1):nrow(forecasts),], 
                                groups = tourism_gts$groups, 
                                keep = 'all'
                        )
                }
                write.csv(
                        na.omit(reconciled_forecasts),
                        file.path(data_dir, out_dir, paste(gsub("\\..*","",filename), "reconciled.csv", sep="-")), 
                        row.names = FALSE
                )
        }
}