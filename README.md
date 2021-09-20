# E-Surveillance-Alert-Classification
E-Surveillance Alert Classification
Business Problem
Description:

Prevent break-ins before they occur using IoT security cameras with built-in computer vision capabilities, reducing the need for human intervention. Automated security to safeguard and alert against threats from intrusion or fire using multi-capability sensors such as vibration, motion, smoke, fire, etc. Ensure the safety of both monetary and intellectual assets with round-the-clock surveillance and controlled access management.

Problem Statement
We are tasked with classifying the alert whether it is Critical, Normal, or Testing which is received from the various sensors. Such as vibration, motion, smoke, fire.

Real world/Business Objectives and Constraints
1.The cost of a mis-classification can be very high.

2.No strict latency concerns.

Mapping the real world problem to an ML problem
Type of Machine Leaning Problem
Supervised Learning:

It is a Multi classification problem, for a given sensor data we need to classify if it is critical, Normal, or Testing

Train and Test Construction
We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.
