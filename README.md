# Astronomical Matching

Collection of algorithms to create matchings between astronomical catalogs.

[![DOI](https://zenodo.org/badge/535167307.svg)](https://zenodo.org/badge/latestdoi/535167307)

## Background

Astronomical photos inherintly have error in them. To best predict where objects are, multiple photos of the same part of the sky are combined. This is equivalent to matching objects in one image to objects in another image. Existing methods rely on nearest-neighbor heuristics which do not take into account the fact that two objects from one image cannot be assigned to the same object.
