(define (problem taxi-problem)
  (:domain taxi)

  (:objects
     taxi1 - taxi
     passenger1 - passenger
     loc-0-0
     loc-0-1
     loc-0-2
     loc-0-3
     loc-0-4
     loc-1-0
     loc-1-1
     loc-1-2
     loc-1-3
     loc-1-4
     loc-2-0
     loc-2-1
     loc-2-2
     loc-2-3
     loc-2-4
     loc-3-0
     loc-3-1
     loc-3-2
     loc-3-3
     loc-3-4
     loc-4-0
     loc-4-1
     loc-4-2
     loc-4-3
     loc-4-4
       - location)

  (:init
     (at taxi1 loc-4-4)
     (at-passenger passenger1 loc-0-4)
     (connected loc-0-0 loc-1-0)
     (connected loc-1-0 loc-0-0)
     (connected loc-0-0 loc-0-1)
     (connected loc-0-1 loc-0-0)
     (connected loc-0-1 loc-1-1)
     (connected loc-1-1 loc-0-1)
     (connected loc-0-1 loc-0-2)
     (connected loc-0-2 loc-0-1)
     (connected loc-0-2 loc-1-2)
     (connected loc-1-2 loc-0-2)
     (connected loc-0-2 loc-0-3)
     (connected loc-0-3 loc-0-2)
     (connected loc-0-3 loc-1-3)
     (connected loc-1-3 loc-0-3)
     (connected loc-0-3 loc-0-4)
     (connected loc-0-4 loc-0-3)
     (connected loc-0-4 loc-1-4)
     (connected loc-1-4 loc-0-4)
     (connected loc-1-0 loc-2-0)
     (connected loc-2-0 loc-1-0)
     (connected loc-1-0 loc-1-1)
     (connected loc-1-1 loc-1-0)
     (connected loc-1-1 loc-2-1)
     (connected loc-2-1 loc-1-1)
     (connected loc-1-1 loc-1-2)
     (connected loc-1-2 loc-1-1)
     (connected loc-1-2 loc-2-2)
     (connected loc-2-2 loc-1-2)
     (connected loc-1-2 loc-1-3)
     (connected loc-1-3 loc-1-2)
     (connected loc-1-3 loc-2-3)
     (connected loc-2-3 loc-1-3)
     (connected loc-1-3 loc-1-4)
     (connected loc-1-4 loc-1-3)
     (connected loc-1-4 loc-2-4)
     (connected loc-2-4 loc-1-4)
     (connected loc-2-0 loc-3-0)
     (connected loc-3-0 loc-2-0)
     (connected loc-2-0 loc-2-1)
     (connected loc-2-1 loc-2-0)
     (connected loc-2-1 loc-3-1)
     (connected loc-3-1 loc-2-1)
     (connected loc-2-1 loc-2-2)
     (connected loc-2-2 loc-2-1)
     (connected loc-2-2 loc-3-2)
     (connected loc-3-2 loc-2-2)
     (connected loc-2-2 loc-2-3)
     (connected loc-2-3 loc-2-2)
     (connected loc-2-3 loc-3-3)
     (connected loc-3-3 loc-2-3)
     (connected loc-2-3 loc-2-4)
     (connected loc-2-4 loc-2-3)
     (connected loc-2-4 loc-3-4)
     (connected loc-3-4 loc-2-4)
     (connected loc-3-0 loc-4-0)
     (connected loc-4-0 loc-3-0)
     (connected loc-3-0 loc-3-1)
     (connected loc-3-1 loc-3-0)
     (connected loc-3-1 loc-4-1)
     (connected loc-4-1 loc-3-1)
     (connected loc-3-1 loc-3-2)
     (connected loc-3-2 loc-3-1)
     (connected loc-3-2 loc-4-2)
     (connected loc-4-2 loc-3-2)
     (connected loc-3-2 loc-3-3)
     (connected loc-3-3 loc-3-2)
     (connected loc-3-3 loc-4-3)
     (connected loc-4-3 loc-3-3)
     (connected loc-3-3 loc-3-4)
     (connected loc-3-4 loc-3-3)
     (connected loc-3-4 loc-4-4)
     (connected loc-4-4 loc-3-4)
     (connected loc-4-0 loc-4-1)
     (connected loc-4-1 loc-4-0)
     (connected loc-4-1 loc-4-2)
     (connected loc-4-2 loc-4-1)
     (connected loc-4-2 loc-4-3)
     (connected loc-4-3 loc-4-2)
     (connected loc-4-3 loc-4-4)
     (connected loc-4-4 loc-4-3)
  )

  (:goal (served passenger1))
)
