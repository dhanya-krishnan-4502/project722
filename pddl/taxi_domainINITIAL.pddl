; (define (domain taxi)
;   (:requirements :strips :typing)

;   (:types taxi passenger location)

;   (:predicates
;     (at ?t ?l)
;     (at-passenger ?p ?l)
;     (in-taxi ?p)
;     (connected ?l1 ?l2)
;     (served ?p)
;   )

;   ;; Move between adjacent locations
;   (:action move
;     :parameters (?t ?from ?to)
;     :precondition (and (at ?t ?from) (connected ?from ?to))
;     :effect (and (not (at ?t ?from)) (at ?t ?to))
;   )

;   ;; Pick up passenger
;   (:action pickup
;     :parameters (?t ?p ?l)
;     :precondition (and (at ?t ?l) (at-passenger ?p ?l))
;     :effect (and (not (at-passenger ?p ?l)) (in-taxi ?p))
;   )

;   ;; Drop off passenger
;   (:action dropoff
;     :parameters (?t ?p ?l)
;     :precondition (and (at ?t ?l) (in-taxi ?p))
;     :effect (and (not (in-taxi ?p)) (at-passenger ?p ?l) (served ?p))
;   )
; )


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

  ;; Drop off passenger
  (:action dropoff
    :parameters (?t - taxi ?p - passenger ?l - location)
    :precondition (and 
        (at ?t ?l)
        (in-taxi ?p)
    )
    :effect (and
        (not (in-taxi ?p))
        (at-passenger ?p ?l)
        (served ?p)
    )
  )
)
