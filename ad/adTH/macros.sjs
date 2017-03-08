// Adapted from: https://github.com/iffsid/ad.js/blob/master/macros/index.js

// -*- mode: js -*-
// precedence and associativity taken from
// http://sweetjs.org/doc/main/sweet.html#custom-operators
operator +   14 { $r } => #{ ad.scalar.add(0, $r) }
operator -   14 { $r } => #{ ad.scalar.sub(0, $r) }
operator *   13 left { $l, $r } => #{ ad.scalar.mul($l, $r) }
operator /   13 left { $l, $r } => #{ ad.scalar.div($l, $r) }
operator %   13 left { $l, $r } => #{ ad.scalar.mod($l, $r) }
operator +   12 left { $l, $r } => #{ ad.scalar.add($l, $r) }
operator -   12 left { $l, $r } => #{ ad.scalar.sub($l, $r) }
operator <   10 left { $l, $r } => #{ ad.scalar.lt($l, $r) }
operator <=  10 left { $l, $r } => #{ ad.scalar.leq($l, $r) }
operator >   10 left { $l, $r } => #{ ad.scalar.gt($l, $r) }
operator >=  10 left { $l, $r } => #{ ad.scalar.geq($l, $r) }
operator ==   9 left { $l, $r } => #{ ad.scalar.eq($l, $r) }
operator !=   9 left { $l, $r } => #{ ad.scalar.neq($l, $r) }
operator ===  9 left { $l, $r } => #{ ad.scalar.peq($l, $r) }
operator !==  9 left { $l, $r } => #{ ad.scalar.pneq($l, $r) }

// TODO? - the pre/post increment/decrement nuance only comes with assignment
//       - requires wrapping up when isolated as statement (x++);
macro ++ {
  rule { $r } => { $r = $r + 1 }
  rule infix { $l | } => { $l = $l + 1 }
}
macro -- {
  rule { $r } => { $r = $r - 1 }
  rule infix { $l | } => { $l = $l - 1 }
}

macro += {
  rule infix { $var:expr | $exprVal:expr } => { $var = $var + $exprVal }
}
macro -= {
  rule infix { $var:expr | $exprVal:expr } => { $var = $var - $exprVal }
}
macro /= {
  rule infix { $var:expr | $exprVal:expr } => { $var = $var / $exprVal }
}
macro *= {
  rule infix { $var:expr | $exprVal:expr } => { $var = $var * $exprVal }
}

macro Math {
  rule { .$x } => { ad.scalar.$x }
}

export +
export -
export *
export /
export <
export <=
export >
export >=
export ==
export !=
export ===
export !==
export ++
export --
export +=
export -=
export /=
export *=
export Math
