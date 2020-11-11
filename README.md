# Convolutional Denosing AutoEncoders - Feature Learning for LISA

Master's Project for my Msc Astronomy & Physics at the University of Glasgow.

Abstract:

"Several uniquely challenging problems will need to be overcome in order to make data analysis for the Laser Interferometer Space Antenna (LISA) possible. First and foremost is the identification of individual sources from a data stream containing an unknown high number of overlapping signals. Source separation is still an unsolved problem in signal processing, and in recent years machine-learning techniques have increasingly been proposed as solutions. In this work we explore the ways in which a feature learning approach could help tackling LISA’s challenges. We develop a machine-learning algorithm using Convolutional Denosing AutoEncoders (CDAEs) and apply it to the separation of overlapping sources within mixed spectrograms. Each CDAE is trained to separate one source while treating the other sources in the mixture as background noise. When tested on unseen data the model achieves excellent results, demonstrating the ability to learn ordered spatial information and successfully and reliably separating the sources present by type. Moving forward, the CDAE model shows the potential to be a solution to the LISA data analysis problems."

The repository contains:

- Pdf of the final project report
- Python scripts of the CDAE model and training routine; jupyter notebooks that apply the model to a variety of different example signals
- A folder of pre-trained models (the results of which were used in the report)
- A folder with outdated code (the most up-to-date code is in the folder "Code")



N.B. One on the example signals used in the report was a simulation of a Massive Black Hole Binary Merger. This simulation was obtained using programs developed by the LISA team, access to which was kindly provided by my supervisor, and, being confidential in nature, have not been included here.
