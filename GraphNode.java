import java.util.*;

public class GraphNode {
   
   // node's internal bias
   private double bias;
   // weights of each incoming edge
   private List<Double> incoming_weights;
   
   // constructs a GraphNode
   public GraphNode (int num_incoming_weights) {
      
      // set the node's internal bias randomly between 0 and 1
      this.bias = Math.random();
      this.incoming_weights = new ArrayList<>();
      
      // for each incoming edge, assign a weight between 0 and 1
      for (int i = 0; i < num_incoming_weights; i++) {
      
         this.incoming_weights.add(Math.random());
      
      }
   
   }
   
   
   // bias get function
   public double getBias() {
   
      return this.bias;
   
   }
   
   // bias set function
   public double setBias(double new_bias) {
   
      this.bias = new_bias;
      return this.bias;
   
   }
   
   // weights get function
   public List<Double> getWeights() {
   
      return this.incoming_weights;
   
   }
   
   // singular weight set function
   public double setWeight(int index, double new_weight) {
   
      this.incoming_weights.set(index, new_weight);
      return this.incoming_weights.get(index);
   
   }
}