# Gene Structure Prediction using Hidden Markov Models (HMM) & Viterbi Algorithm

This project implements a **Hidden Markov Model (HMM)** combined with the **Viterbi Algorithm** to predict gene structures within a given DNA sequence. Specifically, the model identifies:
* **Exons (E)**
* **Introns (I)**
* **5' Splice Sites (5)**

By analyzing a sequence of observed nucleotides (`A`, `C`, `G`, `T`), the algorithm calculates the most probable sequence of hidden biological states that generated them.

Think of this project as a game of tracking a secret agent walking through hidden rooms:
1. **The Hidden States (Rooms):** The agent moves between rooms labeled **Exon**, **Intron**, and **5' Splice Site**. You cannot see which room they are in.
2. **The Emissions (Candies):** Every time the agent takes a step, they drop a nucleotide (`A`, `C`, `G`, or `T`). You *can* see this trail.
3. **The Rules:** Different rooms have different transition rules (e.g., you must pass through the 5' Splice Site to get from an Exon to an Intron) and different nucleotide preferences.

The **Viterbi Algorithm** acts as a smart scoreboard. It steps through the DNA sequence from left to right, calculates the highest probability score for each possible path, discarding low-scoring options along the way. At the end, it traces the winning path backward to output the final predicted gene structure.

## Implementation Steps

1. **Matrix Initialization:** Setting up a `viterbi_value_matrix` to store maximum log-probabilities and a `viterbi_trace_matrix` to remember the path choices.
2. **Node Probability Calculation (`calculate_prob_for_a_node`):** A custom function that computes transition and emission scores for a single cell in log-scale ($+ \text{ instead of } \times$) to avoid numerical underflow.
3. **Matrix Population:** Using nested `for` loops to iterate through all nucleotide positions and hidden states to fill the scoring grids.
4. **Traceback Execution:** Reconstructing the absolute best hidden path from the final sequence position back to the beginning.

