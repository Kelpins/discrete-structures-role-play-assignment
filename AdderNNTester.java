import java.util.*;

public class AdderNNTester {

   public static void main (String[] args) {
      
      // define the number of times our training data will run
      int epochs = 12500;
      
      // define the number of nodes in the hidden layer
      int num_hidden_nodes = 10;
      
      // define the learning rate of our neural network
      double learning_rate = 0.000000195;
      
      // initialize our graph-based neural network with ten nodes in its hidden layer and 1.875 * 10^-7 learning rate
      AdderNeuralNet graph_nn = new AdderNeuralNet(num_hidden_nodes, learning_rate);
      
      // define the train-test split as 80% 20% (arbitrary here but would matter for models with real data)
      double train_pct = 0.8;
      double test_pct = 0.2;
      
      // initialize maps for training and testing data
      Map<ArrayList<Double>, Double> train_data = new HashMap<>();
      Map<ArrayList<Double>, Double> test_data = new HashMap<>();
      
      // propagate the training data with two random input values and their sum as the output value
      for (int i = 0; i < train_pct * 10000; i++) {
         
         // generate two random numbers between 0 and 1000
         double num1 = Math.random() * 1000;
         double num2 = Math.random() * 1000;
         
         // add them to a list
         ArrayList<Double> input = new ArrayList<>();
         input.add(num1);
         input.add(num2);
         
         // determine their sum
         double output = num1 + num2;
         
         // use the list of two numbers and their sum as a key : value pair in our training data
         train_data.put(input, output);
      
      }
      
      // propogate the testing data with two random input values and their sum as the output value
      for (int i = 0; i < test_pct * 10000; i++) {
         
         // generate two random numbers between 0 and 1000
         double num1 = Math.random() * 1000;
         double num2 = Math.random() * 1000;
         
         // add them to a list
         ArrayList<Double> input = new ArrayList<>();
         input.add(num1);
         input.add(num2);
         
         // determine their sum
         double output = num1 + num2;
         
         // use the list of two numbers and their sum as a key : value pair in our testing data
         test_data.put(input, output);
      
      }
      
      // test the neural network on the randomly assigned weights before training the model
      double test_before = graph_nn.test(test_data);
      
      // generate two random numbers 0 - 1000 to run a specific test case
      double num_1_test = Math.random() * 1000;
      double num_2_test = Math.random() * 1000;
      
      // determine their sum to use as the actual answer
      double actual_answer = Math.round((num_1_test + num_2_test) * 1000.0) / 1000.0;
      
      // test the specific case to show the neural network won't work before training
      double test_num_before = graph_nn.testAdd(num_1_test, num_2_test);
      test_num_before = Math.round(test_num_before * 1000) / 1000.0;
      
      
      // train the graph-based neural network with amount of epochs determined above
      graph_nn.train(train_data, epochs);
      
      
      // test the trained neural network on the same test data to verify improvement
      double test_after = graph_nn.test(test_data);
      
      // test the same case as earlier to ensure the neural network works
      double test_num_after = graph_nn.testAdd(num_1_test, num_2_test);
      test_num_after = Math.round(test_num_after * 1000) / 1000.0;
      
      // round off the numbers to three decimal points before printing
      num_1_test = Math.round(num_1_test * 1000) / 1000.0;
      num_2_test = Math.round(num_2_test * 1000) / 1000.0;
      
      
      // print out before/after results to show comparison
      System.out.println("Before Training, Neural Net had an error of: " + test_before);
      System.out.println("After Training, Neural Net had an error of: " + test_after);
      
      System.out.println("Before Training: " + num_1_test + " + " + num_2_test + " = " + test_num_before + ". Actual Answer: " + actual_answer);
      System.out.println("After Training: " + num_1_test + " + " + num_2_test + " = " + test_num_after + ". Actual Answer: " + actual_answer);
      
      
      // use for debugging to visualize weights & biases (using the .toString() function)
      //System.out.println("Before Training, Neural Net appeared: \n" + graph_nn);
      //System.out.println("After Training, Neural Net appeared: \n" + graph_nn);
   
   }

}