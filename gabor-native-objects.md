# Object separation is a well-posed question in a local frame and an ill-posed one in a global one

### A method for recognition, segmentation, and manipulation carried entirely in wave-packet parameter space

*Antti Luode (PerceptionLab), with Claude as implementation collaborator. Do not hype. Do not lie. Just show.*

---

## Abstract

We argue that the choice of image basis decides, before any learning happens, whether "which
part of this image is that object" is even a well-formed question. In a global frequency basis
every coefficient has support over the entire image, so no coefficient belongs to any object and
object membership is undefinable in the coefficient set. In a spatially localized frame — a frame
of oriented wave packets (Gabor atoms) — every atom carries a position, so object membership
becomes a grouping problem over located primitives rather than an impossibility. This is a
statement about representability, not accuracy, and it is the one load-bearing claim of this paper.

On top of that claim we specify **Splatnet**: a pipeline that (1) fits an image with a sparse set of
oriented wave packets by direct per-image optimization, with an explicit price on each atom; (2)
groups the fitted atoms into objects entirely in atom-parameter space, never returning to a pixel
grid or a global spectrum; (3) recognizes objects from the parameter signature of each atom cluster;
and (4) treats segmentation, editing, and cross-frame object permanence as the *same* representation
viewed three ways — because moving an object is literally translating a subset of atoms, and a
persistent object is an atom cluster that survives warm-started re-fitting between frames. We do not
claim to beat modern segmentation networks on segmentation accuracy; mature object-centric systems
already do that. We claim a different property set — one editable, generative, sparsity-priced
substrate on which recognition, separation, and manipulation are a single object — and we state the
failure modes that make this hard, in advance, so the proposal is falsifiable rather than hopeful.

---

## 1. The one claim that carries the paper

Fix an image `I` of `N` pixels. A representation is a set of atoms `{a_k}` and a rendering rule
`I ≈ Σ_k c_k · φ_k`, where `φ_k` is atom `k`'s spatial footprint and `c_k` its coefficient.

**Global frame (Fourier / DCT).** Each `φ_k` is a plane wave (or a block-cosine) whose support is
the whole image (or a fixed block tiling independent of content). The coefficient `c_k` is a single
number describing *all pixels at once*. Ask "does atom `k` belong to the tractor or the background?"
There is no answer: `φ_k` overlaps the tractor and the background and everything else in equal,
content-blind measure. Object membership is not a hard problem in this frame — it is an
**undefined** one. This is why frequency-domain manipulation edits the whole image globally: a
band you suppress is suppressed under every object simultaneously (this is directly observable in a
plain FFT band filter — suppress the fine band and *every* surface loses its texture at once).

**Local frame (oriented wave packets).** Each `φ_k` is a Gaussian-windowed sinusoid pinned to a
position `p_k`, with an orientation `θ_k`, a spatial frequency `f_k`, a phase `ρ_k`, and a size
`σ_k`. Its support is a bounded neighborhood of `p_k`. Now "does atom `k` belong to the tractor?"
is a **well-posed** question: it reduces to whether `p_k` lands on the tractor and whether the
atom's parameters cohere with its neighbors'. The question has moved from *undefined* to *a grouping
problem over located primitives* — which is hard, but hard is a category we can attack.

That is the whole thesis. Everything below is engineering built on the observation that **locality
converts object assignment from ill-posed to well-posed**, and a global frequency basis structurally
cannot offer that.

A necessary honesty: locality makes grouping *possible*, not *solved*. A pile of correctly placed
atoms is not yet a set of objects; the grouping still has to be performed. The rest of this paper is
about doing that grouping in a way that keeps the representation editable and generative throughout.

---

## 2. What "Gabor-native" means precisely

We call a pipeline **atom-native** if every computation after the initial fit consumes
atom-parameter tuples

```
a_k = (p_k, θ_k, f_k, ρ_k, σ_k, amp_k, layer_k)          # ~8 numbers per atom
```

and never a dense pixel grid and never a global spectrum. Rendering to pixels happens exactly once,
at the very end, only if a human wants to look. This is the discipline that buys the editability:
because the entire downstream stack — grouping, recognition, tracking — reads and writes atom
tuples, any operation it can express is an operation a human can also express by hand on the same
tuples, and vice versa. There is no learned latent that only a decoder can interpret. The
representation is legible by construction.

Contrast this with the two nearest existing families, stated fairly:

- **Slot-based object-centric learning** (Slot Attention and its descendants: MONet, GENESIS,
  SLATE, SAVi, and the 2024–2025 slot-diffusion and query-optimization variants). A slot is an
  abstract latent vector that a decoder network turns into a mask or a region. Slots are
  object-centric but *not spatially grounded primitives*: you cannot read a slot's position and
  orientation off its numbers, and you cannot hand-edit a slot without going back through the
  decoder. They solve grouping with a learned attention bottleneck; we solve it over explicit
  located atoms.

- **Object-centric Gaussian splatting** (the 2024–2025 line lifting 2D segmentation masks onto 3D
  Gaussian splats). This is the closest existing thing to what we propose, and we should say so
  plainly. The difference is twofold. Their primitive is an isotropic-ish Gaussian *blob* with no
  carrier — no orientation, frequency, or phase — so it cannot represent oriented texture as a first
  class property; and their object structure is *imported* from an external 2D vision foundation
  model and then lifted, rather than *derived from the primitive frame itself*. A recurring
  complaint in that literature — that foundation-model masks reflect surface-level local similarity
  and do not capture persistent object structure across views — is precisely the gap an atom-native
  grouping is meant to address, because our persistence comes from atom correspondence, not from
  re-segmenting each view.

- **Gabor wavelet networks for object representation** (Krüger & Sommer, 2001, and the older Gabor
  filter-bank recognition literature). The *idea* of Gabor atoms as an object substrate is decades
  old and we claim no priority on it. What is new here is not the atom; it is (a) obtaining the atoms
  by direct per-image optimization with an explicit L0-style price rather than a fixed filter bank,
  and (b) running grouping, recognition, and manipulation natively in atom space as one editable
  representation.

So the novelty is a *combination and a well-posedness framing*, not a new primitive. Said in the
ledger: the components are known; the assembly, and the claim about what the assembly makes
possible, is the contribution.

---

## 3. Why the front end must be direct-fit, not an encoder

A tempting design is a feedforward encoder that maps pixels to atoms in one shot. We have direct,
reproducible evidence against relying on that at the point where per-image adaptation matters.

In prior work on the same packet family (the splatstack / layergate arc), amortized per-image gating
of packets was measured, twice, on synthetic and natural data, and **failed** for a specific,
non-accidental reason: a shared encoder representation that also feeds the reconstruction path can be
manipulated by that path (the reconstruction gradient reshapes the very features the gate reads), and
hard per-image gate decisions get stuck on a one-way door. The dictionary itself was never the
bottleneck — direct per-image optimization fit busy images at high fidelity where the amortized
encoder stalled. The lesson transfers: **the atoms should be obtained per image by optimization**, in
the spirit of how scene-fitting methods (Gaussian splatting) fit each scene rather than amortizing
across scenes.

Concretely, the front end is the direct-fit procedure already validated as an interactive tool:
initialize `M` packets across a few frequency layers, then run a few hundred Adam steps of analytic
gradient descent on reconstruction MSE, with a sparsity price `λ` that lets low-value atoms die. Two
properties make this affordable at inference:

1. **Warm-starting.** For video, each frame is initialized from the previous frame's fitted atoms.
   Because consecutive frames differ little, each frame needs only ~10–20 gradient steps, not a full
   fit. This is the same move that makes per-scene fitting practical, and it does double duty here:
   it *is* the object-permanence mechanism (Section 5).

2. **The price is a real, measurable knob.** Each atom's marginal value — the exact increase in
   reconstruction error if that atom alone is removed — is computable in closed form. Ranking atoms
   by marginal value and pruning below `λ` yields a content-adaptive atom count without a learned
   controller. Atoms defend themselves by being useful.

The output of the front end is a sparse, per-image, physically-meaningful atom set of size `M(I)`,
typically a few hundred atoms at working resolution.

---

## 4. Grouping and recognition, carried out in atom space

Given atoms `{a_k}`, we define an **object** operationally, with no external notion imported:

> An object is a maximal cluster of atoms that is spatially compact, internally coherent in its
> parameters, and temporally persistent under warm-started re-fitting.

### 4.1 Grouping (segmentation)

Build a graph on atoms. Edge weight between `a_i` and `a_j` combines: spatial proximity of
`p_i, p_j`; orientation agreement (continuity of `θ` along contours, i.e. good-continuation between
neighboring packets); frequency-layer compatibility; and amplitude/phase coherence across the shared
neighborhood. Cluster the graph (spectral clustering, or a learned message-passing net that operates
*on atom tuples*, not on pixels). Each cluster is a candidate object; atoms in the low-frequency
layers that spread smoothly across the whole field and cohere with no compact cluster are the
background.

Note what this gives for free that a pixel-mask segmenter does not: the segmentation is
**editable and generative**. A cluster is a set of atoms you can render alone, move, or delete.
There is no separate "cut the object out" step — the object is already a self-contained renderable.

### 4.2 Recognition

Each object cluster has a **parameter signature**: the multiset of its atoms' parameters, made
translation- and scale-invariant by expressing every atom relative to the cluster's centroid and
dominant scale. Recognition is classification of these signatures. Because the signature is a set of
located oriented primitives, natural choices are a set/graph encoder (permutation-invariant over
atoms) or a small transformer over the atom tuples. The classifier never sees pixels; it sees the
arrangement of packets. "These packets, in this relative configuration, are a bicycle."

This is deliberately *analysis-by-synthesis*: the image is explained as a generative arrangement of
atoms, and recognition reads the explanation, rather than a discriminative map from pixels to labels.

### 4.3 Manipulation is not a separate module

Because everything is atom tuples: moving an object is adding a translation to every `p_k` in its
cluster; changing its size is scaling `σ_k` and `p_k` about the centroid; swapping a background is
replacing the background atom set; compositing two objects is concatenating two atom sets and
re-rendering once. Segmentation, recognition, and editing are three questions asked of one
representation, not three networks. This unification is the actual product.

---

## 5. Object permanence, for free, from warm-starting

Because each video frame is fit by warm-starting from the previous frame's atoms, there is a natural
correspondence: atom `k` in frame `t` descends from atom `k` in frame `t−1`. Track the cluster
assignments through that correspondence and an object acquires an identity that persists across
frames even under occlusion-scale changes, as long as enough of its atoms survive. Persistence is
not inferred by a separate tracker re-detecting the object each frame; it is the *default* outcome of
continuity in the fit, and it is exactly the property the object-centric-splatting literature reports
missing when object structure is re-imported per view from an external segmenter.

This yields the "move them at will" property the whole idea is chasing: an object is a labeled,
persistent, self-contained bundle of atoms, so translating it, holding it fixed while the background
moves, or carrying it into a new scene are all single operations on that bundle.

---

## 6. Falsifiable predictions

Stated as bets, each cheap to settle, each able to kill the thesis if it comes back wrong.

- **P1 — Locality is necessary (control).** A grouping pipeline given *global* Fourier/DCT
  coefficients instead of atoms will fail to produce spatially coherent object masks at all, because
  the coefficients carry no position. If a Fourier-coefficient grouping *does* yield coherent object
  masks, the well-posedness argument in Section 1 is wrong. Prediction: it does not.

- **P2 — Within-object motion coherence.** Under warm-started tracking, atoms belonging to the same
  object have more coherent inter-frame motion vectors than atoms drawn from different objects,
  measured as within-cluster vs between-cluster motion-vector variance. If within ≈ between, atoms do
  not track objects and Section 5 fails.

- **P3 — Identity survives background swap.** Recognition of an object from its atom signature alone,
  with all background atoms removed, degrades *less* than a global-feature classifier's accuracy on
  the same object cut from the same scene — because the object's atoms are literally separable from
  the background's, whereas a global feature entangles them. If the atom-native recognizer degrades
  *as much*, the separability claim buys nothing.

- **P4 — The price cliff transfers.** The sparsity price `λ` exhibits a predictable threshold: an
  object's atoms survive pruning exactly while `λ` sits below their marginal value, and the class of
  an object should be recoverable from its signature at any `λ` below that cliff. A measured cliff
  that matches a back-of-envelope from marginal values (as it did on the synthetic packet data)
  supports the economics; a smeared, unpredictable one weakens it.

---

## 7. Failure modes, killed in advance

- **The binding problem is the crux, and it is not solved by locality.** Two adjacent objects with
  similar texture will hand the grouper atoms that are spatially and parametrically continuous across
  the true boundary. Good-continuation edges will bleed one object into the other. Locality makes the
  question askable; it does not answer it. Any honest version of this program lives or dies on the
  grouping step, and we should expect it to fail first on touching, similar-textured objects — the
  same regime that is hard for every segmenter.

- **Amortization is the known trap.** Section 3's evidence says a shared encoder feeding both the
  gate and the reconstruction is manipulable by the reconstruction path. If, to save inference
  compute, the grouper is made to condition on a representation the fit also shapes, expect the same
  lock-picking. Keep the fit and the grouper on separate representations, or pay the direct-fit cost.

- **Resolution and compute are real.** Direct-fitting a few hundred oriented packets per frame is
  cheap at low working resolution and gets expensive as resolution and atom count rise. This proposal
  is honest about being, first, a low-to-mid-resolution instrument. Claims about high-fidelity
  recognition are claims to be *tested*, not assumed; the fitting cost as a function of resolution
  must be reported, not hidden.

- **We are not competing on segmentation accuracy.** Modern promptable segmenters and slot models
  produce excellent masks. If this pipeline is evaluated purely on mask IoU against those, it may
  well lose. That is the wrong axis. The claim is a unified editable generative substrate, and it
  must be evaluated on the properties only it has — controllable composition, hand-editability, and
  persistence-by-construction — not on a metric the incumbents were built to maximize.

---

## 8. Open bets

- **Learned grouping over atom tuples.** A permutation-invariant message-passing net on atoms,
  trained with a small amount of mask supervision, versus the unsupervised spectral clustering
  baseline. Which generalizes to touching objects.
- **What the atom signature actually encodes.** Whether object class is linearly readable from the
  relative-atom signature, and which parameters (orientation histogram? layer occupancy? spatial
  layout?) carry the class information.
- **Compositional generalization.** Whether a recognizer trained on isolated objects recognizes them
  in cluttered scenes without retraining — the property object-centric methods are supposed to buy,
  tested on a substrate where objects are genuinely separable.
- **Higher-resolution fitting.** Whether hierarchical or coarse-to-fine fitting keeps the per-frame
  atom fit affordable as resolution grows, so the "instrument" can become a "system."

---

## 9. The narrow claim, restated

We do not claim a new primitive; oriented wave packets are old. We do not claim to beat segmentation
networks at segmentation; they are very good. We claim exactly this, and only this:

1. In a global frequency basis, object membership of a coefficient is **undefined**; in a local
   wave-packet frame it is a **well-posed grouping problem over located primitives.** This is a
   representability fact, provable by the support argument, independent of any model.

2. Given that, a pipeline that fits an image to sparse priced wave packets and then does *all*
   grouping, recognition, and manipulation in atom-parameter space yields one representation in which
   segmentation, recognition, editing, and cross-frame permanence are the same object rather than
   four — and object permanence in particular falls out of warm-started fitting for free.

3. The proposal is falsifiable at four named points (P1–P4) and expected to fail first at one named
   place (touching similar-textured objects), so it can be checked cheaply and killed honestly.

Everything past that is an experiment we have not yet run, and we will label it that way when we run
it.

---

*Built on the splatstack / layergate packet family and its direct-fit editor. This is a method
proposal; no recognition numbers are claimed until the fits, the grouper, and P1–P4 are run.*
