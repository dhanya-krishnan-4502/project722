(define (domain taxi)
  (:requirements :strips)
  (:predicates
    (taxi-at ?r ?c)
    (passenger-at ?pr ?pc)       ; passenger at row/col
    (in-taxi)
    (dest ?dr ?dc)
  )

  (:action move-north
    :precondition (and)
    :effect (and) ) ; we will ground moves via a higher-level approach or keep simple for Sprint 1

  ; For Sprint 1 (prototype), keep PDDL stubbed if needed.
  ; If pyperplan isn’t ready in time, you can keep HTN as the working planner and mark PDDL “parses”.
)
