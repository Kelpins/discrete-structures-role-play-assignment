import java.util.*;

// feedforward neural networks (FNNs) are graphs organized into "layers"; this class defines one such layer
public class GraphLayer {
   
   // list of nodes in this "layer" of the graph
   private ArrayList<GraphNode> nodes;
   
   // constructs a GraphLayer
   public GraphLayer(int num_nodes, int num_weights_per_node) {
      
      this.nodes = new ArrayList<GraphNode>();
      
      // fills the new GraphLayer with the appropriate amount of nodes/weights
      for (int i = 0; i < num_nodes; i++) {
         
         // puts an instance of the GraphNode class into the layer
         this.nodes.add(new GraphNode(num_weights_per_node));
      
      }
   
   }
   
   // nodes get function
   public ArrayList<GraphNode> getNodes() {
   
      return this.nodes;
   
   }

}