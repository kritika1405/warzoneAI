# warzoneAI
CIS – 667 Introduction to Artificial Intelligence

Project Report For Warzone

Kritika Chugh
SUID : 882046659
SU Email : kchugh@syr.edu


Team Members : Warzone
Mitarth Vaid (mivaid)
Kritika Chugh (kchugh)


Introduction
Warzone is derived from Dameo which is a strategy board game for 2 players, It
was invented by Christian Freeling[6] in 2000. It is a variation of the game draughts
and is played on an 8 by 8 checkered game board. It is considered one of the few
abstract game that rarely ends in a draw[1].
There are 18 pieces per player on the 8 by 8 board initially. Each player’s pieces
are arranged so that the bottom three rows, from the view of each player are filled
with 4,6,8 pawns respectively. Since we have the option of multiple board sizes, we
will have different number of pieces on other board size.
Initially, when the game starts, all the pieces are of same stature but it may change
in the later stages

AI
We created 3 AIs for the game. They include baseline AI, tree-based AI and neural
network AI.
• Baseline AI: This AI takes its decision uniformly at random from the available
choices. The decisions taken from this AI are chosen at random and will be not
take into account the strength of the human user and consequences of the move.
• Tree-Based AI: This AI is based on the minimax algorithm which is a recursive
algorithm generally used in decision-making and game theory. It is used because
it provides an optimal move for the AI assuming that opponent is also playing
optimally. The decisions taken by this AI will take into consideration the move
played by the human and will be aimed to provide better chances at winning
for the AI.
• Neural Network AI: This AI takes its decision based on a combination of
neural network model and monte-carlo tree based algorithm and aims to provide 
the strongest level of AI among all. With further training the model, this AI 
can increase its strength and be more precise, fast and challenging.The decisions
taken from this AI are chosen after performing monte-carlo tree search algorithm
on the available choices given by the neural network model.

Tree based AI
Minimax[7] is a fundamental decision making algorithm which is used in artificial
intelligence, game theory and many more domains such as statistics, philosophy etc.
for minimizing the possible loss for the worst case scenario thereby maximizing gains.
It was originally formulated for n-player zero-sum game theory, which covered both
the cases where players take alternate moves or simultaneous moves. It was later
extended to more complex games and to general decision-making in the presence of
uncertainty.
Minimax guarantees as good an outcome as possible in the worst case scenario.
Consider the case when an opponent makes mistakes that help the AI (a”good case”),
but they can also choose actions very bad for the AI (a ”bad case”), or as bad as
possible for the AI (the ”worst case”). The minimax algorithm will get the best
possible outcome for the AI, assuming the worst case. If the opponent makes a
mistake, then the AI can do even better than the result guaranteed by the minimax
algorithm.
• We score each game from the perspective of the AI, so the AI wants to maximize
the final game score.
• We have positive final scores are good for the AI and bad for the opponent and
negative final scores that are bad for the AI and good for the opponent.
• AI is called the ”max” player and human is called the ”min” player.
The basic structure[3] of the algorithm is given by the following function:-
minimax(node):
return score for max(node)
child results = []
2
for child in children(node):
child results.append(minimax(child))
if player(node) == ”max”: result = max(child results)
if player(node) == ”min”: result = min(child results)
return result

Neural Networks
Neural networks, also called Artificial neural networks[5] are computational systems based on the interpretation of biological neural networks that constitute animal
brains. A Neural Network is constructed on collection of nodes called neurons, which
resemble the neurons in a biological brain. Each connection can transmit a signal to
other neurons just like the synapses in brain. An artificial neuron which receives a
signal, processes it, and then signals it to the neurons connected to it. The connections are called edges and the signal at each connection is a real number. The output
of each neuron is calculated by employing a non-linear function of its inputs. All
the neurons and edges have a weight that adjusts as the model learns. The weight
increases or decreases the strength of the signal at a connection. A neural network
model is trained to reduce the mean squared error to 0. Usually, neurons are aggregated into a set of layers but if the model is not to be complex, a single layer can
work. Distinct layers perform distinct transformations on their inputs. Signals travel
from the first layer (the input layer), to the last layer (the output layer) giving the
final output. I used one hot encoding to encode my state into valid neural network
model and then use convolutional neural network[2] to operate upon it



