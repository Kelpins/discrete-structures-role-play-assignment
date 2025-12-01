import java.util.*;

public class AdderNeuralNet {
   
   // initialize the network's one hidden layer
   private GraphLayer hidden_layer;
   
   // output layer defined as a singular node, because our adder should return one double
   private GraphNode output_node;
   
   // initialize the learning rate
   private double learning_rate;
   
   
   // ANN constructor function
   public AdderNeuralNet(int hidden_layer_size, double learning_rate) {
      
      // number of incoming weights to each hidden layer node is two, one for each node in the input layer
      this.hidden_layer = new GraphLayer(hidden_layer_size, 2);
      
      // number of incoming weights is equal to the size of the hidden layer, as each node represents one incoming edge
      this.output_node = new GraphNode(hidden_layer_size);
      
      // define the learning rate (a scalar changing how much each step of backpropogation affects each bias/weight)
      this.learning_rate = learning_rate;
   
   }
   
   // activation function of choice (ReLU)
   private double ReLUActivation (double input) {
      
      // rectified linear unit activation returns 0 if negative, or the input if positive
      return Math.max(0, input);
         
   }
   
   // feedforward function runs values through the neural network calculating an output
   private double[] feedForward (ArrayList<Double> inputs, double expected_output) {
      
      // gather the hidden nodes into an ArrayList for ease of access during our feedforward calculation
      ArrayList<GraphNode> hidden_nodes = this.hidden_layer.getNodes();
      
      // define a variable for the number of hidden nodes for ease of access during our feedforward calculation
      int num_hidden_nodes = hidden_nodes.size();
      
      // an array storing activation function outputs for each hidden node, with a final index reserved for error
      // we need to store the hidden activations as they are important for our backpropagation calculations
      double[] hidden_activations = new double[num_hidden_nodes+1];
      
      // initialize variable for the feedforward-calculated output 
      double actual_output = 0.0;
      // initialize variable for the running error calculation
      double error = 0.0;
      
      // loop through every hidden node in the layer
      for (int i = 0; i < num_hidden_nodes; i++) {
         
         // for each hidden node, loop through for each input node
         for (int j = 0; j < inputs.size(); j++) {
            
            // begin the running activation calculation for that hidden node (input 1 * weight 1 + input 2 * weight 2)
            hidden_activations[i] += inputs.get(j) * hidden_nodes.get(i).getWeights().get(j);
                  
         }
         
         // add the bias of each hidden node to its own running calculation
         hidden_activations[i] += hidden_nodes.get(i).getBias();
         // run the final activation function on each running calculation
         hidden_activations[i] = ReLUActivation(hidden_activations[i]);
      
      }
      
      // use the hidden activations from this single step of feedforward to determine the output of the step
      for (int i = 0; i < hidden_activations.length-1; i++) {
         
         // begin the running activation calculation for the output step (activation 1 * weight 1 + ... + activation n * weight n)
         actual_output += hidden_activations[i] * this.output_node.getWeights().get(i);
      
      }
      
      // add the bias of the output node to the running calculation
      actual_output += this.output_node.getBias();
      // run the ouput through the ReLU activation function one final time
      actual_output = ReLUActivation(actual_output);
      
      // error is the derivative of our cost function, and is calculated by expected value - calculated output
      error = expected_output - actual_output;
      
      // place error in the final, reserved slot of our hidden activations list so future functions can access it as well
      hidden_activations[hidden_activations.length-1] = error;
            
      // return the hidden layer activations and error
      return hidden_activations;
   
   }
   
   // backpropagation is the next step in training a feedforward neural network
   private void backPropogation (ArrayList<Double> key, double[] activations_and_error) {
      
      //retreive the error from the reserved spot of the activations and error list
      double error = activations_and_error[activations_and_error.length-1];
      
      // begin the backpropagation at the end; change the bias of the output node in the direction of the error
      this.output_node.setBias(this.output_node.getBias() + (this.learning_rate * error));
      
      // going back through each node of the hidden layer and affecting the weights an biases proportional to the learning rate
      for (int i = 0; i < activations_and_error.length-1; i++) {
         
         // first change each edge coming into the output node
         this.output_node.setWeight(i, this.output_node.getWeights().get(i) + this.learning_rate * error * activations_and_error[i]);
         // then change the bias of each node in the hidden layer
         GraphNode node = this.hidden_layer.getNodes().get(i);
         node.setBias(node.getBias() + learning_rate * error * this.output_node.getWeights().get(i));
         
         // finally change the weight of each edge between the input layer and the hidden layer
         for (int j = 0; j < key.size(); j++) {
         
            node.setWeight(j, node.getWeights().get(j) + this.learning_rate * error * this.output_node.getWeights().get(i) * key.get(j));
         
         }
               
      }
   
   }
   
   
   // define the train function to run feedforward and backpropagation over the entire training dataset for each epoch
   public void train (Map<ArrayList<Double>, Double> data, int epochs) {
      
      // train over the course of e epochs
      for (int e = 0; e < epochs; e++) {
         
         // cost keeps track of the running error or the epoch; the overall mean squared error
         double cost = 0.0;
         
         // stochastic gradient descent runs a full feedforward and backpropagation for each data point
         for (ArrayList<Double> key : data.keySet()) {
            
            // run feedforward on each datapoint
            double[] activations_and_error = feedForward(key, data.get(key));
            
            // increment the cost by the MSE
            cost += Math.pow(activations_and_error[activations_and_error.length-1], 2)/2;
            
            // run backpropogation given the activation values outputted by feedforward
            backPropogation(key, activations_and_error);
         
         }
         
         // print each epoch completion with associated cost value as a progress indicator
         System.out.println("Epoch " + (e+1) + "/" + epochs + " completed with cost " + cost/data.size() + ".");
      
      }
   
   }
   
   
   // testing function only runs feedforward, as it will change no weights/biases; only test the current ones
   public double test (Map<ArrayList<Double>, Double> data) {
      
      // initialize a running cost which evaluates the efficacy of the model
      double cost = 0.0;
      
      // run through each datapoint in the testing dataset
      for (ArrayList<Double> key : data.keySet()) {
         
         // for each datapoint, run it through feedforward
         double[] activations_and_error = feedForward(key, data.get(key));
         
         // since the test function doesnt run backpropagation, the error output is all that matters.
         cost += Math.pow(activations_and_error[activations_and_error.length-1], 2)/2;
                  
      }
      
      // divide by the size of the dataset to produce the MSE (mean squared error); integral of the error function
      cost /= data.size();
      
      return cost;
   
   }
   
   
   // another test function to evaluate the efficacy of our model; this one simulates the model's implementation, providing an output for a specific case with no expected output
   public double testAdd (double num1, double num2) {
      
      // put the two inputs into an ArrayList, which is the datatype the model accepts
      ArrayList<Double> input = new ArrayList<>();
      input.add(num1);
      input.add(num2);
      
      // uses 0 as expected output because feedforward calculates expected - actual, therefore this function provides the network's output, but negative
      double[] activations_and_error = feedForward(input, 0);
      // subtract from 0 to make the result the model's actual answer
      return (0 - activations_and_error[activations_and_error.length-1]);
     
   }
   
   // the tostring function allows this class to be included in print statements
   public String toString() {
      
      // create a reference to hidden nodes for ease of access in the string below
      ArrayList<GraphNode> hidden_nodes = this.hidden_layer.getNodes();
   
      // for use in debugging; will not print the whole model but gives a small sample of the biases and weights to ensure they are within reason
      return "bias a: " + hidden_nodes.get(0).getBias() + "; weights a: " + hidden_nodes.get(0).getWeights() + "\n" + "bias b: " + hidden_nodes.get(1).getBias() + "; weights b: " + hidden_nodes.get(1).getWeights() + "\n" + "bias c: " + hidden_nodes.get(2).getBias() + "; weights c: " + hidden_nodes.get(2).getWeights() + "\n" + "bias o: " + this.output_node.getBias() + "; weights o: " + this.output_node.getWeights();
   
   }

}