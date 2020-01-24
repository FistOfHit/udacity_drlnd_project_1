# Project 1: Navigation - Hitesh Kumar

# Unity Banana Navigation

Note: Code was adapted from working in the provided workspace to working on local, but wasnt tested on local. If in doubt, just run it in the workspace and it should be fine, provided you change the banaa path back to normal and add the python installation line (`!pip -q install ./python`)

## Project details

This is my submission for the first project in the Udacity DRL nanodegree.

#### Environment and state details

The environment is provided to us in the form of a simulation run in Unity. It starts with the agent, able to see with a certain FOV angle, a room with yellow and blue bananas around it. The purpose is to train the agent to collect yellow bananas and avoid blue bananas. The environment is considered solved when the Agent can collect an average of 13 bananas over 100 consecutive episodes.

The state is a 37 dimensional array with information on the position of the agent and the Bananas around it. In any given state the agent has 4 possible actions (up, down, turn left and right)

## Solution
Episode: 0, Average Score: 0.00

Episode: 50, Average Score: 0.69

Episode: 100, Average Score: 1.01

Episode: 150, Average Score: 2.14

Episode: 200, Average Score: 4.97

Episode: 250, Average Score: 7.33

Episode: 300, Average Score: 7.23

Episode: 350, Average Score: 7.01

Episode: 400, Average Score: 8.18

Episode: 450, Average Score: 9.39

Episode: 500, Average Score: 11.45

Environment solved in 531 episodes! 
Average Score: 13.03

## Getting started

The requirements are included in `requirements.txt`, so using `conda install requirements.txt` should be enough to install everything. Of course, you should also follow the instructions provided in the original repository by the Udacity team, and also make sure you have unity installed as per the instructions provided in the first tab in the navigation project lesson.

## Instructions

Simply either run the cells in `solution.ipynb` or if you prefer run the script `solution.py`. If you really want to play around with the hyperparameters, you are free to do so in the files provided, but this is not reccommended.
