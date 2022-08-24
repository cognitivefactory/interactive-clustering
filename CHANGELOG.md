# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


<!-- insertion marker -->
## [0.5.3](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.5.3) - 2022-08-24

<small>[Compare with 0.5.2](https://github.com/cognitivefactory/interactive-clustering/compare/0.5.2...0.5.3)</small>

### Build
- refactor scripts and configs ([1f56aeb](https://github.com/cognitivefactory/interactive-clustering/commit/1f56aeb6c13eaad67c520bf67272078bdc4a63c2) by SCHILD Erwan).


## [0.5.2](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.5.2) - 2022-08-22

<small>[Compare with 0.5.1](https://github.com/cognitivefactory/interactive-clustering/compare/0.5.1...0.5.2)</small>

### Code Refactoring
- update copier-pdm template ([bd93764](https://github.com/cognitivefactory/interactive-clustering/commit/bd937640d388ef8345937c15ea2505b881e38ce8) by SCHILD Erwan).


## [0.5.1](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.5.1) - 2022-02-16

<small>[Compare with 0.5.0](https://github.com/cognitivefactory/interactive-clustering/compare/0.5.0...0.5.1)</small>

### Bug Fixes
- update constraints manager serialization ([6111542](https://github.com/cognitivefactory/interactive-clustering/commit/61115428e949ee1f51e3d9d2a3d42c6a0ef4c48b) by Erwan Schild).

### Code Refactoring
- speed up constraints transitivity ([5e255ef](https://github.com/cognitivefactory/interactive-clustering/commit/5e255eff67f3c8371343bf15c14bdda7079ddc69) by Erwan Schild).
- update spacy usage ([2d05289](https://github.com/cognitivefactory/interactive-clustering/commit/2d0528919da69535a845820a33999f6e42c2036c) by Erwan Schild).


## [0.5.0](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.5.0) - 2022-02-15

<small>[Compare with 0.4.2](https://github.com/cognitivefactory/interactive-clustering/compare/0.4.2...0.5.0)</small>

### Bug Fixes
- force nb_cluster to be smaller than dataset size ([981fd0f](https://github.com/cognitivefactory/interactive-clustering/commit/981fd0fb6486da97579fd094bae6b6bec2d2f3cc) by Erwan Schild).
- correct kmeans centroid computation for deleted data ids ([b0cd1a0](https://github.com/cognitivefactory/interactive-clustering/commit/b0cd1a081112f40e7a18c18ba48141ddc6f9c2b1) by Erwan Schild).

### Code Refactoring
- update to python 3.7 ([f363998](https://github.com/cognitivefactory/interactive-clustering/commit/f3639985f074db26df56702d5584b60b63b20943) by Erwan Schild).
- make format ([f363998](https://github.com/cognitivefactory/interactive-clustering/commit/f3639985f074db26df56702d5584b60b63b20943) by Erwan Schild).

### Features
- add constraints manager serialization ([c2e13e2](https://github.com/cognitivefactory/interactive-clustering/commit/c2e13e2a903a493cf30b1ecc82e332051a32a47c) by Erwan Schild).


## [0.4.2](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.4.2) - 2021-10-27

<small>[Compare with 0.4.1](https://github.com/cognitivefactory/interactive-clustering/compare/0.4.1...0.4.2)</small>

### Bug Fixes
- force csr_matrix for spacy vectors in order to perform vstack ([be7b75c](https://github.com/cognitivefactory/interactive-clustering/commit/be7b75ca9678d068777de60efc4d2ef4aaa3d11f) by Erwan Schild).

### Code Refactoring
- refactor code and force sparse matrix ([63e94a2](https://github.com/cognitivefactory/interactive-clustering/commit/63e94a2ff6d0d270d9de3267d2eea2d82aa5b117) by Erwan Schild).
- speed up spectral clustering ([711cf4d](https://github.com/cognitivefactory/interactive-clustering/commit/711cf4db3173f0c7fa1f93a464a590176db3c2ea) by Erwan Schild).


## [0.4.1](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.4.1) - 2021-10-19

<small>[Compare with 0.4.0](https://github.com/cognitivefactory/interactive-clustering/compare/0.4.0...0.4.1)</small>

### Code Refactoring
- speed up clustering and refactor code ([93ada28](https://github.com/cognitivefactory/interactive-clustering/commit/93ada28045d8d386ec5756abe6d97a22ab973716) by Erwan Schild).


## [0.4.0](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.4.0) - 2021-10-12

<small>[Compare with 0.3.0](https://github.com/cognitivefactory/interactive-clustering/compare/0.3.0...0.4.0)</small>

### Bug Fixes
- correct networkx dependency requirement ([4ec4587](https://github.com/cognitivefactory/interactive-clustering/commit/4ec45874b14f1b43c57d1b75af31096515beae49) by Erwan SCHILD).
- correct networkx import in sampling ([521a4ff](https://github.com/cognitivefactory/interactive-clustering/commit/521a4ff35862d7fa8efc08f0d8c1d87597c5b395) by Erwan Schild).
- speed up computation of sampling.clusters_based.sampling for distance restrictions ([5ab6821](https://github.com/cognitivefactory/interactive-clustering/commit/5ab68219bd41ff09581be65e6505a7f6d77a02b4) by Erwan Schild).
- speed up computation of constraints.binary.get_min_and_max_number_of_clusters ([1e50f7c](https://github.com/cognitivefactory/interactive-clustering/commit/1e50f7c32b2353c62102d0cbd19748991845b408) by Erwan Schild).

### Code Refactoring
- update template with copier update ([e0a7c77](https://github.com/cognitivefactory/interactive-clustering/commit/e0a7c776426694d618a1c7eba518fba9d861f02e) by Erwan Schild).
- fix black dependenciy installation ([eef88c5](https://github.com/cognitivefactory/interactive-clustering/commit/eef88c598757edb9c90b8fbe104b059632cce80e) by Erwan Schild).
- delete old random sampler ([6cd0a06](https://github.com/cognitivefactory/interactive-clustering/commit/6cd0a06ea94fc7b3377e6f1e15e82a8a36f0d0ef) by Erwan Schild).

### Features
- implementation of getter of data IDs involved in a constraint conflict ([6eace0d](https://github.com/cognitivefactory/interactive-clustering/commit/6eace0d0dd0ba4068769f7152119e8fbb1ee90cb) by Erwan Schild).


## [0.3.0](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.3.0) - 2021-10-04

<small>[Compare with 0.2.1](https://github.com/cognitivefactory/interactive-clustering/compare/0.2.1...0.3.0)</small>

### Features
- update Constraints Sampling with clusters/distance/known_constraints restrictions ([34c5747](https://github.com/cognitivefactory/interactive-clustering/commit/34c57475c9b6e20cc8bbdcefe9b58b0d962f0042) by Erwan Schild).


## [0.2.1](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.2.1) - 2021-09-20

<small>[Compare with 0.2.0](https://github.com/cognitivefactory/interactive-clustering/compare/0.2.0...0.2.1)</small>

### Bug Fixes
- correct constraints transitivity inconsistencies in BinaryConstraintsManager.add_constraint ([98f162e](https://github.com/cognitivefactory/interactive-clustering/commit/98f162e5785df13d99e25f757be4988e5fab757c) by Erwan Schild).

### Code Refactoring
- fix code quality errors ([02c03ee](https://github.com/cognitivefactory/interactive-clustering/commit/02c03ee4fb1cf50a28c896c93a43ee00708a2d38) by Erwan Schild).
- update exception message ([2003a1e](https://github.com/cognitivefactory/interactive-clustering/commit/2003a1e782a2885ae45419577aa9d599f08876a3) by Erwan Schild).


## [0.2.0](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.2.0) - 2021-09-01

<small>[Compare with 0.1.3](https://github.com/cognitivefactory/interactive-clustering/compare/0.1.3...0.2.0)</small>

### Bug Fixes
- change constraints storage from sorted lists to sets ([47d3528](https://github.com/cognitivefactory/interactive-clustering/commit/47d35284deda6ae2e26b4fb87170d96a599fcba3) by Erwan Schild).

### Code Refactoring
- delete utils.checking ([a9a1f50](https://github.com/cognitivefactory/interactive-clustering/commit/a9a1f50ba0d101e5a212e825f907a567f64e05f7) by Erwan Schild).
- remove checks and force usage of constraints_manager ([4cdb0bb](https://github.com/cognitivefactory/interactive-clustering/commit/4cdb0bbd11fdababb35fc4a48612897274bc69b8) by Erwan Schild).
- improve sampling speed ([9d6ed5c](https://github.com/cognitivefactory/interactive-clustering/commit/9d6ed5c1b4baeddd35e9bbb3e32c4fbc74031633) by Erwan Schild).
- add py.typed file ([25c7be3](https://github.com/cognitivefactory/interactive-clustering/commit/25c7be3648ff9dabf84b1349f2ae52df4ba4c8ae) by Erwan Schild).


## [0.1.3](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.1.3) - 2021-05-20

<small>[Compare with 0.1.2](https://github.com/cognitivefactory/interactive-clustering/compare/0.1.2...0.1.3)</small>


## [0.1.2](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.1.2) - 2021-05-19

<small>[Compare with 0.1.1](https://github.com/cognitivefactory/interactive-clustering/compare/0.1.1...0.1.2)</small>

### Code Refactoring
- correct format and tests ([e3245f3](https://github.com/cognitivefactory/interactive-clustering/commit/e3245f33bf09680d926bdd890c162f6e4df1ab4d) by Erwan SCHILD).


## [0.1.1](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.1.1) - 2021-05-18

<small>[Compare with 0.1.0](https://github.com/cognitivefactory/interactive-clustering/compare/0.1.0...0.1.1)</small>


## [0.1.0](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.1.0) - 2021-05-17

<small>[Compare with first commit](https://github.com/cognitivefactory/interactive-clustering/compare/8e0d8b45342caf2850a9238ddba48c9b7f1d7f44...0.1.0)</small>

### Bug Fixes
- fix encoding error on fr_core_news_sm-2.3.0/meta.json ? ([98acb42](https://github.com/cognitivefactory/interactive-clustering/commit/98acb42d3fc0e06fe6687d28f3f4b531243a88ab) by Erwan SCHILD).
- correct installation sources ([de4c727](https://github.com/cognitivefactory/interactive-clustering/commit/de4c727798f174fbd0c87ae6f6204b09a8d0a131) by Erwan SCHILD).

### Code Refactoring
- order import and update documentation ([70e8780](https://github.com/cognitivefactory/interactive-clustering/commit/70e8780749f9a5b8cc0ae261c5f33e4712480dad) by Erwan SCHILD).
- remove local fr_core_news_sm model ([1f9da8f](https://github.com/cognitivefactory/interactive-clustering/commit/1f9da8fee5fc9728114e19753a54a7e4edacc25d) by Erwan SCHILD).
- test fr_core_news_sm installation like a pip package ([b249159](https://github.com/cognitivefactory/interactive-clustering/commit/b2491599b1ff0e606f449fa37208e0a1c31484b4) by Erwan SCHILD).

### Features
- implement Interactive Clustering ([d678d87](https://github.com/cognitivefactory/interactive-clustering/commit/d678d87d0d485cb7b977bb229b9bb8c6fc590c66) by Erwan SCHILD).
