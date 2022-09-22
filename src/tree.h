#include<iostream>
#include<Rcpp.h>
using namespace std;
using namespace Rcpp;

int sample_int(int n){
  return rand() % n;
}

// Creating one sequence of values
NumericVector seq_along_cpp(int n){
  NumericVector vec_seq;
  for(int i =0; i<n;i++){
    vec_seq.push_back(i);
  }
  return vec_seq;
}

class node{

  // Storing parameters
public:
  int index; // Storing the node index
  NumericVector obs_train; // Storing train observations index that belong to that node
  NumericVector obs_test; // Storing test observations index that belong to that node
  int left; // Index of the left node
  int right; // Index of the right node
  int parent; // Index of the parent
  int depth; //  Depth of the node

  int var; // Variable which the node was split
  double var_split; // Variable which the node was split

  double mu; // Mu parameter from the node
  double loglike; // Loglikelihood of the residuals in that terminal node
public:

  // Getting the constructor
  node(int index_n, NumericVector obs_train_numb, NumericVector obs_test_numb,
       int left_i, int right_i, int depth_i, int var_i, double var_split_i,
       double mu_i){

    index = index_n;
    obs_train = obs_train_numb;
    obs_test = obs_test_numb;
    left = left_i;
    right = right_i;
    depth = depth_i;
    var = var_i;
    var_split = var_split_i;
    mu = mu_i;
  }

  void DisplayNode(){

    cout << "Node Number: " << index << endl;
    cout << "Decision Rule -> Var:  " << var << " & Rule: " << var_split << endl;
    cout << "Left <-  " << left << " & Right -> " << right << endl;

    if(true){
      cout << "Observations train: " ;
      for(int i = 0; i<obs_train.size(); i++){
        cout << obs_train[i] << " ";
      }
      cout << endl;
    }

    if(true){
      cout << "Observations test: " ;
      for(int i = 0; i<obs_test.size(); i++){
        cout << obs_test[i] << " ";
      }
      cout << endl;
    }

  }

  bool isTerminal(){
    return ((left == -1) && (right == -1) );
  }

  double loglikelihood(Rcpp::NumericVector residuals_values,
                       double tau,
                       double tau_mu){
    // Declaring quantities in the node
    int n_train = obs_train.size();
    double sum_r_sq = 0;
    double sum_r = 0;

    for(int i = 0; i<n_train;i++){
      sum_r_sq =  sum_r_sq + residuals_values(obs_train(i))*residuals_values(obs_train(i));
      sum_r = sum_r + residuals_values(obs_train(i));
    }

    return -0.5*tau*sum_r_sq -0.5*log(tau_mu+(n_train*tau)) + (0.5*(tau*tau)*(sum_r*sum_r))/( (tau*n_train)+tau_mu );
  }

  void update_mu(Rcpp::NumericVector residuals_values,
                   double tau,
                   double tau_mu){
    // Calculating the sum of residuals
    double sum_r = 0;
    double n_train = residuals_values.size();

    for(int i = 0; i<obs_train.size();i++){
      sum_r+=residuals_values(obs_train(i));
    }

    mu = R::rnorm(sum_r/(tau*n_train+tau_mu),1/sqrt(n_train*tau+tau_mu));

  }


};


class Tree{

public:
  // Defining the main element of the tree structure
  vector<node> list_node;

  // Getting the vector of nodes
  Tree(int n_obs_train,int n_obs_test){
    // Creating a root node
    list_node.push_back(node(0,
                             seq_along_cpp(n_obs_train),
                             seq_along_cpp(n_obs_test),
                             -1, // left
                             -1, // right
                             0, //depth
                             -1, // var
                             -1.1, // var_split
                             0 )); // loglike
  }

  // void DisplayNodesNumber(){
  //   cout << "The tree has " << list_node.size() << " nodes" << endl;
  // }

  void DisplayNodes(){
    for(int i = 0; i<list_node.size(); i++){
      list_node[i].DisplayNode();
    }
    cout << "# ====== #" << endl;
  }


  // Getting terminal nodes
  vector<node> getTerminals(){

    // Defining terminal nodes
    vector<node> terminalNodes;

    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==1 ){ // Check this again, might remove the condition of being greater than 5
        terminalNodes.push_back(list_node[i]); // Adding the terminals to the list
      }
    }
    return terminalNodes;
  }

  // Getting terminal nodes
  vector<node> getNonTerminals(){

    // Defining terminal nodes
    vector<node> NonTerminalNodes;

    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==0){
        NonTerminalNodes.push_back(list_node[i]); // Adding the terminals to the list
      }
    }
    return NonTerminalNodes;
  }


  // Getting the number of n_terminals
  int n_terminal(){

    // Defining the sum value
    int terminal_sum = 0;
    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==1){
        terminal_sum++;
      }
    }

    return terminal_sum;
  }

  // Getting the number of non-terminals
  int n_internal(){

    // Defining the sum value
    int internal_sum = 0;
    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==0){
        internal_sum++;
      }
    }

    return internal_sum;
  }

  // Get the number of NOG (branches parents of terminal nodes)
  int n_nog(){
    int nog_counter = 0;
    for(int i=0;i<list_node.size();i++){
      if(list_node[i].isTerminal()==0){
        if(list_node[list_node[i].left].isTerminal()==1 && list_node[list_node[i].right].isTerminal()==1){
          nog_counter++;
        }
      }
    }
    return nog_counter;
  }

  // Growing a tree
  void grow(Rcpp::NumericMatrix x_train,
            Rcpp::NumericMatrix x_test,
            int node_min_size,
            Rcpp::NumericMatrix xcut){

    // Defining the number of covariates
    int p = x_train.ncol();
    int cov_trial_counter = 0; // To count if all the covariate were tested
    int valid_split_indicator = 0;
    int n_train,n_test;
    int g_original_index;
    int split_var;
    int max_index;
    int g_index;
    double min_x_current_node;
    double max_x_current_node;


    Rcpp::NumericVector x_cut_valid; // Defining the vector of valid "split rules" based on xcut and the terminal node
    Rcpp::NumericVector x_cut_candidates;
    node* g_node; // Node to be grow

    // Select a terminal node to be randomly selected to split
    vector<node> candidate_nodes = getTerminals();
    int n_t_node ;

    // Find a valid split indicator within a valid node
    while(valid_split_indicator==0){


            // Getting the number of terminal nodes
            n_t_node = candidate_nodes.size();

            // Getting a NumericVector of all node index
            Rcpp::NumericVector all_node_index;

            // Selecting a random terminal node
            g_index = sample_int(candidate_nodes.size());

            // Iterating over all nodes

            for(int i=0;i<list_node.size();i++){

              all_node_index.push_back(list_node[i].index); // Adding the index into the list

              if(list_node[i].index==candidate_nodes[g_index].index){
                g_original_index = i;
              }
            }


            // Getting the maximum index in case to build the new indexes for the new terminal nodes
            max_index = max(all_node_index);

            // Selecting the terminal node
            g_node = &candidate_nodes[g_index];

            // Selecting the random rule
            split_var = sample_int(p);

            // Getting the number of train
            n_train = g_node->obs_train.size();
            n_test = g_node->obs_test.size();

            // Selecting the column of the x_curr
            Rcpp::NumericVector x_current_node ;

            for(int i = 0; i<g_node->obs_train.size();i++) {
              x_current_node.push_back(x_train(g_node->obs_train(i),split_var));
            }

            min_x_current_node = min(x_current_node);
            max_x_current_node = max(x_current_node);

            // Getting available xcut variables
            x_cut_candidates = xcut(_,split_var);

            // Create a vector of splits that will lead to nontrivial terminal nodes
            for(int k = 0; k<x_cut_candidates.size();k++){
              if(x_cut_candidates(k)>min_x_current_node && x_cut_candidates(k)<max_x_current_node){
                x_cut_valid.push_back(x_cut_candidates(k));
              }
            }

            // Do not grow a tree and skip it or select a new tree
            if(x_cut_valid.size()==0){

              cov_trial_counter++;
              // CHOOSE ANOTHER SPLITING RULE
                if(cov_trial_counter == p){
                      return;
                }
            } else {

            // Verifying that a valid split was selected
            valid_split_indicator = 1;

            }
    }

    // Sample a g_var
    double split_var_rule = x_cut_valid(sample_int(x_cut_valid.size()));

    // Creating the new terminal nodes based on the new xcut selected value
    // Getting observations that are on the left and the ones that are in the right
    Rcpp::NumericVector new_left_train_index;
    Rcpp::NumericVector new_right_train_index;
    Rcpp::NumericVector curr_obs_train; // Observations that belong to that terminal node

    // Same from above but for test observations
    // Getting observations that are on the left and the ones that are in the right
    Rcpp::NumericVector new_left_test_index;
    Rcpp::NumericVector new_right_test_index;
    Rcpp::NumericVector curr_obs_test; // Observations that belong to that terminal node

    /// Iterating over the train and test
    for(int j=0; j<n_train;j++){
      if(x_train(g_node->obs_train(j),split_var)<=split_var_rule){
        new_left_train_index.push_back(g_node->obs_train(j));
      } else {
        new_right_train_index.push_back(g_node->obs_train(j));
      }
    }

    /// Iterating over the test and test
    for(int i=0; i<n_test;i++){
      if(x_test(g_node->obs_test(i),split_var)<=split_var_rule){
        new_left_test_index.push_back(g_node->obs_test(i));
      } else {
        new_right_test_index.push_back(g_node->obs_test(i));
      }
    }

    // Updating the current node
    list_node[g_original_index].left = max_index+1;
    list_node[g_original_index].right = max_index+2;
    list_node[g_original_index].var = split_var;
    list_node[g_original_index].var_split = split_var_rule;

    // Updating the current_tree
    list_node.insert(list_node.begin()+g_original_index+1,
                     node(max_index+1,
                             new_left_train_index,
                             new_left_test_index,
                             -1,
                             -1,
                             g_node->depth+1,
                             -1,
                             -1.1,
                             0));

    list_node.insert(list_node.begin()+g_original_index+2,
                     node(max_index+2,
                             new_right_train_index,
                             new_right_test_index,
                             -1,
                             -1,
                             g_node->depth+1,
                             -1,
                             -1.1,
                             0));

    // Finishing the process;
    return;

  }

  // Creating the verb to prune a tree
  void prune(){

    // Selecting possible parents of terminal nodes to prune
    vector<node> nog_list;
    int nog_original_index;
    // Getting the parent of terminal node only
    for(int i = 0; i<list_node.size();i++){
        if(list_node[i].terminal==0){
          if(list_node[i+1].terminal == 1 && list_node[i+2].terminal==1){
            nog_list.push_back();
          }
        }
    }

  }
};



class test{
  public:

  int x;
  Rcpp::NumericVector x_vec;

  test(int number){
    x = number;
    x_vec = seq_along_cpp(number);
  }

  Rcpp::NumericVector double_x(){

    Rcpp::NumericVector double_x_vector;
    for(int i = 0; i< x_vec.size(); i++){
      double_x_vector.push_back(2*x_vec(i));
    }
    return double_x_vector;
  }

  double sum_double_x(){
    double sum = 0;
    Rcpp::NumericVector x_vector_dummy = double_x();
      for(int i = 0;i<x_vec.size(); i++){
        sum += x_vector_dummy(i);
      }
    return sum;
  }
};

RCPP_EXPOSED_CLASS(test);
RCPP_EXPOSED_CLASS(node);
RCPP_EXPOSED_CLASS(Tree);
