## Model

- 1 way
- 2 lanes
- N drivers, 5-6 different types of drivers
  - From less to more aggresive (less to more prob. of taking risky decissions, log-norm distributed? test.)
  - Continuous space, discrete time
  - N vehicles (consequently to N drivers)

## Considerations

- Accordion effect
- Driver's behavior (depends on the type of driver, considerations about distance, speed, etc.)
- Genetic algorithm to find the best parameters for the drivers' behavior
  - Take N prev. drivers to generate the next driver
  - Simulate N drivers from the newly introduced and evaluate (do this separately in a pure virtual environment)
- Observe which driver of the N gets the best results
- Evaluate which type of driver/drivers is/are the best for the global system 