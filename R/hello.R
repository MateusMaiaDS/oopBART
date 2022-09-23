# # Creating a simple example to test the loglikelihood function
# library(purrr)
# n <- 20
# x <- matrix(seq(-1,10,length.out = n))
# y <- 10*x
#
# test_likelihood(x = x,y = y,tau = 10,tau_mu = 10)
#
# # Dummy function to calculte loglikelihood from one node
# dummy_loglikelihood <- function(resid,
#                                 tau,tau_mu){
#   sum_resid_sq <- sum(resid^2)
#   sum_resid <- sum(resid)
#   n_train <- length(resid)
#   return(-0.5*log(tau_mu+n*tau)-0.5*tau*sum_resid_sq+0.5*((tau^2)*(sum_resid^2))/(tau_mu*n_train+tau))
# }
#
#
# dummy_loglikelihood(resid = y,tau = 10,tau_mu = 10)
#
# # Exampe for the update of mu
# dummy_update_mu <- function(resid,
#                             tau,
#                             tau_mu){
#   return(rnorm(n = 1,mean = sum(resid)/(tau*length(resid)+tau_mu),sd = 1/sqrt((tau*length(resid)+tau_mu))))
# }
# rcpp_samples <- replicate(test_mu_update(x = x,y = y,tau = 10,tau_mu = 10),n = 1000)
# r_samples <- replicate(dummy_update_mu(resid = y,tau = 10,tau_mu = 10),n = 1000)
#
# hist(rcpp_samples, col = rgb(1,0,0,0.5))
# hist(r_samples,col = rgb(0,0,1,0.5), add =T )
#
#
# # Creating a test for the grow verb
# num_cut <- 100
# # Cut matrix
# numcut <- num_cut
# xcut <- matrix(NA,ncol = ncol(x),nrow = numcut)
#
# # Getting possible x values
# for(j in 1:ncol(x)){
#   xs <- quantile(x[ , j], type=7,
#                  probs=(0:(numcut+1))/(numcut+1))[-c(1, numcut+2)]
#
#   xcut[,j] <-xs
# }
#
#
# indicator_vec <- test_change(x = x,x_test = x,y = c(y),xcut = xcut,tau = 10,tau_mu = 10)
#
# df <- data.frame(x = x, y = y , col = as.character(indicator_vec))
# df$col %>% unique() %>% length()
#
# library(ggplot2)
# ggplot(df)+
#   geom_point(mapping = aes(x = x, y = y, col = col))+
#   geom_vline(xintercept = xcut)
#
