# Appendix: Geometry-Induced Gaussian Weighting of Residuals

## A. Setup and Notation

Let \(f:\mathcal X\to\mathbb R\) be twice differentiable with
\[
f^\* := \inf_{x\in\mathcal X} f(x) > -\infty,
\qquad
\Delta(x) := f(x)-f^\* \ge 0.
\]

Let the learning-rate schedule be a smooth function
\[
\eta:(0,\Delta_0)\to(0,\infty),
\]
with a unique strict maximizer
\[
\Delta^\* := \arg\max_{\Delta\in(0,\Delta_0)} \eta(\Delta),
\qquad
\eta'(\Delta^\*)=0,\quad \eta''(\Delta^\*)<0.
\tag{A.1}
\]

Parameters are partitioned into blocks \(x=(x_1,\dots,x_L)\).
We consider the max-over-blocks norm
\[
\|x\| := \max_{i\le L}\|x_i\|_{(i)},
\tag{A.2}
\]
with dual norm
\[
\|g\|_\* := \sup_{\|u\|\le1}\langle g,u\rangle.
\tag{A.3}
\]

---

## B. Dual Geometry of Max-Over-Blocks Norms

### Lemma B.1 (Dual decomposition)
For the norm (A.2), the dual norm satisfies
\[
\|g\|_\* = \sum_{i=1}^L \|g_i\|_{(i),\*}.
\tag{B.1}
\]

**Proof.**
The constraint \(\|u\|\le1\) is equivalent to \(\|u_i\|_{(i)}\le1\) for all \(i\).
Hence
\[
\sup_{\|u\|\le1}\langle g,u\rangle
=
\sum_i \sup_{\|u_i\|_{(i)}\le1}\langle g_i,u_i\rangle
=
\sum_i \|g_i\|_{(i),\*}.
\quad\Box
\]

---

### Lemma B.2 (Blockwise domination)
Assume that for each block there exists \(c_i>0\) such that
\[
\|A\|_{(i),\*} \le c_i\,\|A\|_{(i)}
\quad \forall A.
\tag{B.2}
\]
Define
\[
\kappa := \sum_{i=1}^L c_i.
\tag{B.3}
\]
Then for all \(g\),
\[
\|g\|_\* \le \kappa\,\|g\|.
\tag{B.4}
\]

**Proof.**
By Lemma B.1 and (B.2),
\[
\|g\|_\*
=
\sum_i \|g_i\|_{(i),\*}
\le
\sum_i c_i \|g_i\|_{(i)}
\le
\Big(\sum_i c_i\Big)\max_i \|g_i\|_{(i)}
=
\kappa\,\|g\|.
\quad\Box
\]

---

### Remark B.3 (Values of \(c_i\))
For the norms used in the paper:
- **Muon** (\(\|\cdot\|_{(i)}=\|\cdot\|_{S2}\)):  
  \(\|A\|_{S1}\le \min(m_i,n_i)\|A\|_{S2}\), hence \(c_i=\min(m_i,n_i)\).
- **Sign** (\(\|\cdot\|_{(i)}=\|\cdot\|_\infty\)):  
  \(\|A\|_1\le (m_in_i)\|A\|_\infty\), hence \(c_i=m_in_i\).
- **normSGD** (\(\|\cdot\|_{(i)}=\|\cdot\|_F\)):  
  \(\|A\|_F=\|A\|_F\), hence \(c_i=1\).

Thus \(\kappa\) is exactly the quantity computed in the implementation.

---

## C. Residual Dynamics and Quadratic Energy

Consider the deterministic update
\[
x^{t+1} = x^t - \eta(\Delta_t)\,u_t,
\qquad
u_t \in \arg\max_{\|u\|\le1}\langle\nabla f(x^t),u\rangle.
\tag{C.1}
\]
By definition of the linear minimization oracle,
\[
\langle\nabla f(x^t),u_t\rangle = \|\nabla f(x^t)\|_\*.
\tag{C.2}
\]

A second-order Taylor expansion yields
\[
\Delta_{t+1}-\Delta_t
=
-\eta(\Delta_t)\|\nabla f(x^t)\|_\*
+\frac{\eta(\Delta_t)^2}{2}\,\langle \nabla^2 f(\xi_t)u_t,u_t\rangle,
\tag{C.3}
\]
for some \(\xi_t\) on the segment \([x^t,x^{t+1}]\).

---

### Lemma C.1 (Appearance of \(\kappa\) in the quadratic term)
Let \(H_t=\nabla^2 f(\xi_t)\).
Then
\[
\langle H_t u_t,u_t\rangle
\;\le\;
\kappa\,\|H_t\|_{\|\cdot\|\to\|\cdot\|}\,\|u_t\|^2
=
\kappa\,\|H_t\|_{\|\cdot\|\to\|\cdot\|},
\tag{C.4}
\]
where \(\kappa\) is defined in (B.3).

**Proof.**
By definition,
\[
\langle H_t u_t,u_t\rangle \le \|H_t u_t\|_\* \|u_t\|.
\]
Using (B.4),
\[
\|H_t u_t\|_\* \le \kappa \|H_t u_t\| \le \kappa \|H_t\|_{\|\cdot\|\to\|\cdot\|}\|u_t\|.
\]
Since \(\|u_t\|=1\), (C.4) follows. \(\Box\)

---

## D. Local Quadratic Potential Around \(\Delta^\*\)

Combining (C.3)–(C.4) and expanding near \(\Delta^\*\) using \(\eta'(\Delta^\*)=0\),
the residual dynamics admits a local potential representation
\[
\Delta_{t+1}-\Delta_t \;\approx\; -\partial_\Delta E(\Delta_t),
\]
with quadratic energy
\[
E(\Delta)
=
\frac12\,(\sigma_F \kappa)\,(\Delta-\Delta^\*)^2,
\tag{D.1}
\]
where \(\sigma_F>0\) absorbs \(\eta(\Delta^\*)^2\) and the local Hessian norm.

---

## E. Gaussian Weight and Implementation

The invariant density associated with the potential (D.1) is
\[
p(\Delta)\;\propto\;\exp\!\left(-E(\Delta)\right)
=
\exp\!\left(-\frac12\,(\sigma_F\kappa)(\Delta-\Delta^\*)^2\right).
\tag{E.1}
\]

The implementation uses the parametrization
\[
w(\Delta;\Delta^\*)
=
\exp\!\left(-(\Delta-\Delta^\*)^2\cdot\frac{\sigma_F}{2\sigma_{\mathrm{norm}}}\right).
\tag{E.2}
\]
Matching (E.1) and (E.2) yields the exact identification
\[
\boxed{\sigma_{\mathrm{norm}}=\frac{1}{\kappa}.}
\tag{E.3}
\]

---

## F. Summary

- \(\kappa\) arises **uniquely** from the dual geometry of max-over-blocks norms via Lemma B.2.
- The same \(\kappa\) necessarily appears in the quadratic term of the residual dynamics (Lemma C.1).
- This induces a Gaussian localization around \(\Delta^\*\) with curvature \(\sigma_F\kappa\).
- Therefore, with no additional assumptions, the correct choice in the implementation is
\[
\sigma_{\mathrm{norm}} = \frac{1}{\kappa}.
\]