(define (domain taxi)
  (:requirements :strips :typing)

  (:types 
      taxi 
      passenger 
      location
  )

  (:predicates
    (at ?t - taxi ?l - location)
    (at-passenger ?p - passenger ?l - location)
    (in-taxi ?p - passenger)
    (connected ?l1 - location ?l2 - location)
    (goal-loc ?p - passenger ?l - location)   ; NEW
    (served ?p - passenger)
  )

  ;; Move between adjacent locations
  (:action move
    :parameters (?t - taxi ?from - location ?to - location)
    :precondition (and 
        (at ?t ?from)
        (connected ?from ?to)
    )
    :effect (and 
        (not (at ?t ?from)) 
        (at ?t ?to)
    )
  )

  ;; Pick up passenger
  (:action pickup
    :parameters (?t - taxi ?p - passenger ?l - location)
    :precondition (and 
        (at ?t ?l)
        (at-passenger ?p ?l)
    )
    :effect (and 
        (not (at-passenger ?p ?l)) 
        (in-taxi ?p)
    )
  )

  ;; Drop off passenger at their goal location*
  (:action dropoff
    :parameters (?t - taxi ?p - passenger ?l - location)
    :precondition (and 
        (at ?t ?l)
        (in-taxi ?p)
        (goal-loc ?p ?l)          
    )
    :effect (and
        (not (in-taxi ?p))
        (at-passenger ?p ?l)
        (served ?p)
    )
  )
)
