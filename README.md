# Reinforcement Learning for Inventory Management

## Overview
This coursework focuses on **inventory optimization using Reinforcement Learning**, in which participants will train an agent to identify the most effective order-placement strategy.

## Environment
Participants are in charge of a three-echelon supply chain like the one depicted below.
<p align="center">
  <img src=".\SCstructure.png" alt="SupplyChainStructure" width="500"/>
</p>

### Description 
The **supplier** is an oil producer company, whose exquisite bottles are sold in different stores around the city. Due to the distance between the production facilities and the stores, the company owns a **distribution centre (DC)** in the vicinity of the city. **Retailers** sell the product directly to the customers and place replenishment orders to maintain sufficient stock levels. Analogously, the inventory level at the DC must be enough to supply the stores and, hence, DC places replenishment orders directly to the manufacturing company. The challenge is to develop a re-order policy for each participant, since each stage faces uncertain in the demand of the stage succeeding it.

This environment operates under the following assumptions:
- Customer demand is modeled as a random variable following a Poisson distribution.
- Production facilities have immediate access to an unlimited supply of raw materials.

Considering a time horizon of 4 weeks, at each day or time step $t$, the following sequence of events occurs:
1.  DC and retailers place replenishment orders.
2.  DC and retailers receive orders after the corresponding lead time from their respective suppliers and update both inventory on-hand and pipeline inventory.
3.  Each stage satisfies demand of their respective clients according to current inventory levels. 
    1.  Backlogged sales take priority over the orders arriving at current period $t$.
    2.  Then, the orders placed by the retailers at the current period are fulfilled with the remaining available inventory.
    3.  Finally, the backlog of each retailer is updated.
4.  Profit is evaluated as the difference between the sales revenue and the different costs across the entire supply chain (i.e., delivery fees, variable order costs, holding cost, unfulfilled demand penalties and excess capacity cost).

To find the optimal policy via RL, the inventory management problem is modelled as a Markov Decision Process (MDP) characterized by the following elements:
* **Action space:** the agent must decide the number of units each retailer or DC reorders at each time step.
* **State space:** states represent the inventory position at each time step, which is the difference between the total inventory and the backlog.
* **Reward:** the agent tries to maximize the profit of the supply chain.

## Task

In this coursework, participants will learn how to optimize the inventory control policy using tabular Q-learning.  
- Instructions can be found in this **Jupyter Notebook**:  
    📄 [`coursework_Almería.ipynb`](./coursework_Almería.ipynb)  
- Note that only the following Python modules are allowed:
    * numpy
    * time
    * tqdm

## Submissions
* Participants will have to submit their code through the following Streamlit app:
[this link](https://rl-inventory-submission.streamlit.app/).
* The deadline for submitting the code is Friday 19th at 11:00 A.M. 
* Benchmarking will be based on each group's final submission.

## Support
❓ Need help? Do not hesitate to ask instructors during sessions!