// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// bart
List bart(const Rcpp::NumericMatrix x_train, const Rcpp::NumericVector y, const Rcpp::NumericMatrix x_test, const Rcpp::NumericMatrix xcut, int n_tree, int n_mcmc, int n_burn, int n_min_size, double tau, double mu, double tau_mu, double naive_sigma, double alpha, double beta, double a_tau, double d_tau);
RcppExport SEXP _oopBART_bart(SEXP x_trainSEXP, SEXP ySEXP, SEXP x_testSEXP, SEXP xcutSEXP, SEXP n_treeSEXP, SEXP n_mcmcSEXP, SEXP n_burnSEXP, SEXP n_min_sizeSEXP, SEXP tauSEXP, SEXP muSEXP, SEXP tau_muSEXP, SEXP naive_sigmaSEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP a_tauSEXP, SEXP d_tauSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type x_train(x_trainSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type x_test(x_testSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type xcut(xcutSEXP);
    Rcpp::traits::input_parameter< int >::type n_tree(n_treeSEXP);
    Rcpp::traits::input_parameter< int >::type n_mcmc(n_mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type n_burn(n_burnSEXP);
    Rcpp::traits::input_parameter< int >::type n_min_size(n_min_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type tau_mu(tau_muSEXP);
    Rcpp::traits::input_parameter< double >::type naive_sigma(naive_sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type a_tau(a_tauSEXP);
    Rcpp::traits::input_parameter< double >::type d_tau(d_tauSEXP);
    rcpp_result_gen = Rcpp::wrap(bart(x_train, y, x_test, xcut, n_tree, n_mcmc, n_burn, n_min_size, tau, mu, tau_mu, naive_sigma, alpha, beta, a_tau, d_tau));
    return rcpp_result_gen;
END_RCPP
}
// test_grow_prune_method
void test_grow_prune_method(Rcpp::NumericMatrix x, Rcpp::NumericMatrix x_test, Rcpp:: NumericVector y, Rcpp:: NumericMatrix xcut, double tau, double tau_mu);
RcppExport SEXP _oopBART_test_grow_prune_method(SEXP xSEXP, SEXP x_testSEXP, SEXP ySEXP, SEXP xcutSEXP, SEXP tauSEXP, SEXP tau_muSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type x_test(x_testSEXP);
    Rcpp::traits::input_parameter< Rcpp:: NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp:: NumericMatrix >::type xcut(xcutSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type tau_mu(tau_muSEXP);
    test_grow_prune_method(x, x_test, y, xcut, tau, tau_mu);
    return R_NilValue;
END_RCPP
}
// test_grow_prune_structure
void test_grow_prune_structure(Rcpp::NumericMatrix x, Rcpp::NumericMatrix x_test, Rcpp:: NumericVector y, Rcpp:: NumericMatrix xcut, double tau, double tau_mu);
RcppExport SEXP _oopBART_test_grow_prune_structure(SEXP xSEXP, SEXP x_testSEXP, SEXP ySEXP, SEXP xcutSEXP, SEXP tauSEXP, SEXP tau_muSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type x_test(x_testSEXP);
    Rcpp::traits::input_parameter< Rcpp:: NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp:: NumericMatrix >::type xcut(xcutSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type tau_mu(tau_muSEXP);
    test_grow_prune_structure(x, x_test, y, xcut, tau, tau_mu);
    return R_NilValue;
END_RCPP
}
// sum_vec
void sum_vec();
RcppExport SEXP _oopBART_sum_vec() {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    sum_vec();
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_oopBART_bart", (DL_FUNC) &_oopBART_bart, 16},
    {"_oopBART_test_grow_prune_method", (DL_FUNC) &_oopBART_test_grow_prune_method, 6},
    {"_oopBART_test_grow_prune_structure", (DL_FUNC) &_oopBART_test_grow_prune_structure, 6},
    {"_oopBART_sum_vec", (DL_FUNC) &_oopBART_sum_vec, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_oopBART(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
