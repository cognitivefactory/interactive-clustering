# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


<!-- insertion marker -->
## [0.5.3](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.5.3) - 2022-08-25

<small>[Compare with 0.5.2](https://github.com/cognitivefactory/interactive-clustering/compare/0.5.2...0.5.3)</small>

### Build
- add .gitignore ([34a71c9](https://github.com/cognitivefactory/interactive-clustering/commit/34a71c9012acf2a3bc7dfb1e516204d416e4aa7b) by Erwan SCHILD).
- update pyproject.toml with url dependencies ([d063637](https://github.com/cognitivefactory/interactive-clustering/commit/d063637e2268b1faabf24e79ca3bb0d263025a76) by Erwan SCHILD).
- refactor scripts and configs ([1f56aeb](https://github.com/cognitivefactory/interactive-clustering/commit/1f56aeb6c13eaad67c520bf67272078bdc4a63c2) by Erwan SCHILD).


## [0.5.2](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.5.2) - 2022-08-22

<small>[Compare with 0.5.1](https://github.com/cognitivefactory/interactive-clustering/compare/0.5.1...0.5.2)</small>

### Code Refactoring
- update copier-pdm template ([bd93764](https://github.com/cognitivefactory/interactive-clustering/commit/bd937640d388ef8345937c15ea2505b881e38ce8) by Erwan SCHILD).


## [0.5.1](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.5.1) - 2022-02-16

<small>[Compare with 0.5.0](https://github.com/cognitivefactory/interactive-clustering/compare/0.5.0...0.5.1)</small>

### Bug Fixes
- update constraints manager serialization ([6111542](https://github.com/cognitivefactory/interactive-clustering/commit/61115428e949ee1f51e3d9d2a3d42c6a0ef4c48b) by Erwan SCHILD).

### Build
- remove previous dependency fix for python 3.10 ([6509506](https://github.com/cognitivefactory/interactive-clustering/commit/65095061c1cbbbc36441804aa2257cc021cf549d) by Erwan SCHILD).
- update dependencies for python 3.10 ([07e8cad](https://github.com/cognitivefactory/interactive-clustering/commit/07e8cade1d50c7c37b0996c86eea9bdb9ff1a9d9) by Erwan SCHILD).
- update dependencies ([4cc63af](https://github.com/cognitivefactory/interactive-clustering/commit/4cc63af2719d96da3919a52c8254d9a4da8e972d) by Erwan SCHILD).

### Code Refactoring
- speed up constraints transitivity ([5e255ef](https://github.com/cognitivefactory/interactive-clustering/commit/5e255eff67f3c8371343bf15c14bdda7079ddc69) by Erwan SCHILD).
- update spacy usage ([2d05289](https://github.com/cognitivefactory/interactive-clustering/commit/2d0528919da69535a845820a33999f6e42c2036c) by Erwan SCHILD).


## [0.5.0](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.5.0) - 2022-02-15

<small>[Compare with 0.4.2](https://github.com/cognitivefactory/interactive-clustering/compare/0.4.2...0.5.0)</small>

### Bug Fixes
- force nb_cluster to be smaller than dataset size ([9a5a3f0](https://github.com/cognitivefactory/interactive-clustering/commit/9a5a3f0d5611d31a8f622385d3239eff336b80ce) by Erwan SCHILD).
- correct kmeans centroid computation for deleted data ids ([b0cd1a0](https://github.com/cognitivefactory/interactive-clustering/commit/b0cd1a081112f40e7a18c18ba48141ddc6f9c2b1) by Erwan SCHILD).

### Build
- remove python 3.11 from ci ([71768f0](https://github.com/cognitivefactory/interactive-clustering/commit/71768f0423aa438c123a74b4e6b734f4d2458c4c) by Erwan SCHILD).
- update to python 3.7 ([981fd0f](https://github.com/cognitivefactory/interactive-clustering/commit/981fd0fb6486da97579fd094bae6b6bec2d2f3cc) by Erwan SCHILD).
- update .gitignore ([76d8ff8](https://github.com/cognitivefactory/interactive-clustering/commit/76d8ff88cc20082a88ee597f034f189f04870bb0) by Erwan SCHILD).

### Code Refactoring
- make format ([f363998](https://github.com/cognitivefactory/interactive-clustering/commit/f3639985f074db26df56702d5584b60b63b20943) by Erwan SCHILD).

### Features
- add constraints manager serialization ([c2e13e2](https://github.com/cognitivefactory/interactive-clustering/commit/c2e13e2a903a493cf30b1ecc82e332051a32a47c) by Erwan SCHILD).

## [0.4.2](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.4.2) - 2021-10-27

<small>[Compare with 0.4.1](https://github.com/cognitivefactory/interactive-clustering/compare/0.4.1...0.4.2)</small>

### Bug Fixes
- force csr_matrix for spacy vectors in order to perform vstack ([be7b75c](https://github.com/cognitivefactory/interactive-clustering/commit/be7b75ca9678d068777de60efc4d2ef4aaa3d11f) by Erwan SCHILD).

### Build
- remove 'regex' dependency correction ([e448515](https://github.com/cognitivefactory/interactive-clustering/commit/e448515b5e64e46848b2a1d3c6dc5e299ea13d8d) by Erwan SCHILD).

### Code Refactoring
- refactor code and force sparse matrix ([63e94a2](https://github.com/cognitivefactory/interactive-clustering/commit/63e94a2ff6d0d270d9de3267d2eea2d82aa5b117) by Erwan SCHILD).
- speed up spectral clustering ([711cf4d](https://github.com/cognitivefactory/interactive-clustering/commit/711cf4db3173f0c7fa1f93a464a590176db3c2ea) by Erwan SCHILD).


## [0.4.1](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.4.1) - 2021-10-19

<small>[Compare with 0.4.0](https://github.com/cognitivefactory/interactive-clustering/compare/0.4.0...0.4.1)</small>

### Code Refactoring
- speed up clustering and refactor code ([93ada28](https://github.com/cognitivefactory/interactive-clustering/commit/93ada28045d8d386ec5756abe6d97a22ab973716) by Erwan SCHILD).


## [0.4.0](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.4.0) - 2021-10-12

<small>[Compare with 0.3.0](https://github.com/cognitivefactory/interactive-clustering/compare/0.3.0...0.4.0)</small>

### Bug Fixes
- correct networkx dependency requirement ([4ec4587](https://github.com/cognitivefactory/interactive-clustering/commit/4ec45874b14f1b43c57d1b75af31096515beae49) by Erwan SCHILD).
- correct networkx import in sampling ([521a4ff](https://github.com/cognitivefactory/interactive-clustering/commit/521a4ff35862d7fa8efc08f0d8c1d87597c5b395) by Erwan SCHILD).
- speed up computation of sampling.clusters_based.sampling for distance restrictions ([5ab6821](https://github.com/cognitivefactory/interactive-clustering/commit/5ab68219bd41ff09581be65e6505a7f6d77a02b4) by Erwan SCHILD).
- speed up computation of constraints.binary.get_min_and_max_number_of_clusters ([1e50f7c](https://github.com/cognitivefactory/interactive-clustering/commit/1e50f7c32b2353c62102d0cbd19748991845b408) by Erwan SCHILD).

### Code Refactoring
- update template with copier update ([e0a7c77](https://github.com/cognitivefactory/interactive-clustering/commit/e0a7c776426694d618a1c7eba518fba9d861f02e) by Erwan SCHILD).
- fix black dependenciy installation ([eef88c5](https://github.com/cognitivefactory/interactive-clustering/commit/eef88c598757edb9c90b8fbe104b059632cce80e) by Erwan SCHILD).
- delete old random sampler ([6cd0a06](https://github.com/cognitivefactory/interactive-clustering/commit/6cd0a06ea94fc7b3377e6f1e15e82a8a36f0d0ef) by Erwan SCHILD).

### Features
- implementation of getter of data IDs involved in a constraint conflict ([6eace0d](https://github.com/cognitivefactory/interactive-clustering/commit/6eace0d0dd0ba4068769f7152119e8fbb1ee90cb) by Erwan SCHILD).


## [0.3.0](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.3.0) - 2021-10-04

<small>[Compare with 0.2.1](https://github.com/cognitivefactory/interactive-clustering/compare/0.2.1...0.3.0)</small>

## [0.2.1](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.2.1) - 2021-09-20

<small>[Compare with 0.2.0](https://github.com/cognitivefactory/interactive-clustering/compare/0.2.0...0.2.1)</small>

### Bug Fixes
- correct constraints transitivity inconsistencies in BinaryConstraintsManager.add_constraint ([98f162e](https://github.com/cognitivefactory/interactive-clustering/commit/98f162e5785df13d99e25f757be4988e5fab757c) by Erwan SCHILD).

### Build
- fix cvxopt 1.2.7 dependency error for linux ([a2f5429](https://github.com/cognitivefactory/interactive-clustering/commit/a2f54296db651d91617f1c2ff3729b84c8ede22a) by Erwan SCHILD).
- update flake config ([696fb97](https://github.com/cognitivefactory/interactive-clustering/commit/696fb972854e9c4eb27286df97ab5bfd0ecbda50) by Erwan SCHILD).

### Code Refactoring
- fix code quality errors ([02c03ee](https://github.com/cognitivefactory/interactive-clustering/commit/02c03ee4fb1cf50a28c896c93a43ee00708a2d38) by Erwan SCHILD).
- update exception message ([2003a1e](https://github.com/cognitivefactory/interactive-clustering/commit/2003a1e782a2885ae45419577aa9d599f08876a3) by Erwan SCHILD).


## [0.2.0](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.2.0) - 2021-09-01

<small>[Compare with 0.1.3](https://github.com/cognitivefactory/interactive-clustering/compare/0.1.3...0.2.0)</small>

### Bug Fixes
- change constraints storage from sorted lists to sets ([47d3528](https://github.com/cognitivefactory/interactive-clustering/commit/47d35284deda6ae2e26b4fb87170d96a599fcba3) by Erwan SCHILD).

### Build
- update python cross-versions dependencies ([dc4ac32](https://github.com/cognitivefactory/interactive-clustering/commit/dc4ac328e475045d5addf6dba39f34aebd566865) by Erwan SCHILD).
- fix python test dependencies ([8b4106f](https://github.com/cognitivefactory/interactive-clustering/commit/8b4106f8eddb673ccbb394eac2c206447cab99f4) by Erwan SCHILD).
- fix python dependencies ([4e61a57](https://github.com/cognitivefactory/interactive-clustering/commit/4e61a57d0b9c6eb2cb84b5b73b642a56150ba270) by Erwan SCHILD).
- update python dependencies ([8cd3e49](https://github.com/cognitivefactory/interactive-clustering/commit/8cd3e491616fd2ad26ac130a06b79e2a464a84eb) by Erwan SCHILD).
- correct build steps ([68091d4](https://github.com/cognitivefactory/interactive-clustering/commit/68091d4375bcc7e5bf9f5fc86e429ee9e3bdc6bc) by Erwan SCHILD).
- update project from poetry to pdm ([3133391](https://github.com/cognitivefactory/interactive-clustering/commit/313339131c405830d3a941a69176127ae6e4257d) by Erwan SCHILD).
- update .gitignore with migration from poetry to pdm ([31d2374](https://github.com/cognitivefactory/interactive-clustering/commit/31d237440de8d97a0b2bd19d3ad497a63ab07646) by Erwan SCHILD).
- change template informations ([7314555](https://github.com/cognitivefactory/interactive-clustering/commit/73145559e98f383c3b2a64384fd744fe04e2e9a7) by Erwan SCHILD).
- prepare migration from poetry to pdm ([89a214e](https://github.com/cognitivefactory/interactive-clustering/commit/89a214e5270e5a048961ca8fb0858196c4031cc9) by Erwan SCHILD).

### Code Refactoring
- delete utils.checking ([a9a1f50](https://github.com/cognitivefactory/interactive-clustering/commit/a9a1f50ba0d101e5a212e825f907a567f64e05f7) by Erwan SCHILD).
- remove checks and force usage of constraints_manager ([4cdb0bb](https://github.com/cognitivefactory/interactive-clustering/commit/4cdb0bbd11fdababb35fc4a48612897274bc69b8) by Erwan SCHILD).
- improve sampling speed ([9d6ed5c](https://github.com/cognitivefactory/interactive-clustering/commit/9d6ed5c1b4baeddd35e9bbb3e32c4fbc74031633) by Erwan SCHILD).
- add py.typed file ([25c7be3](https://github.com/cognitivefactory/interactive-clustering/commit/25c7be3648ff9dabf84b1349f2ae52df4ba4c8ae) by Erwan SCHILD).


## [0.1.3](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.1.3) - 2021-05-20

<small>[Compare with 0.1.2](https://github.com/cognitivefactory/interactive-clustering/compare/0.1.2...0.1.3)</small>


## [0.1.2](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.1.2) - 2021-05-19

<small>[Compare with 0.1.1](https://github.com/cognitivefactory/interactive-clustering/compare/0.1.1...0.1.2)</small>

### Build
- remove spacy language model as direct dependencies ([95093a6](https://github.com/cognitivefactory/interactive-clustering/commit/95093a6ee7d0875eea19684178d83cff38687c7b) by Erwan SCHILD).
- remove local spacy language model dependencies ([779f737](https://github.com/cognitivefactory/interactive-clustering/commit/779f737fd9df461fd988bad78789eafca049d189) by Erwan SCHILD).
- install fr_core_news_sm as a local dependency ([564fa5c](https://github.com/cognitivefactory/interactive-clustering/commit/564fa5ccb0c3289aaa00719abd87815fc632b460) by Erwan SCHILD).
- add .gitattributes ([b35a1e2](https://github.com/cognitivefactory/interactive-clustering/commit/b35a1e2e2b5759bb0cff2c82cc35d768bb248633) by Erwan SCHILD).

### Code Refactoring
- correct format and tests ([e3245f3](https://github.com/cognitivefactory/interactive-clustering/commit/e3245f33bf09680d926bdd890c162f6e4df1ab4d) by Erwan SCHILD).


## [0.1.1](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.1.1) - 2021-05-18

<small>[Compare with 0.1.0](https://github.com/cognitivefactory/interactive-clustering/compare/0.1.0...0.1.1)</small>

### Build
- correct install source ([c84bb4c](https://github.com/cognitivefactory/interactive-clustering/commit/c84bb4c065b70f28dddc1746d9016f5f1aede581) by Erwan SCHILD).
- correct package import ([4121954](https://github.com/cognitivefactory/interactive-clustering/commit/4121954f4e9dab37551b59a9f108ba8e675f01db) by Erwan SCHILD).

## [0.1.0](https://github.com/cognitivefactory/interactive-clustering/releases/tag/0.1.0) - 2021-05-17

<small>[Compare with first commit](https://github.com/cognitivefactory/interactive-clustering/compare/8e0d8b45342caf2850a9238ddba48c9b7f1d7f44...0.1.0)</small>

### Bug Fixes
- fix encoding error on fr_core_news_sm-2.3.0/meta.json ? ([98acb42](https://github.com/cognitivefactory/interactive-clustering/commit/98acb42d3fc0e06fe6687d28f3f4b531243a88ab) by Erwan SCHILD).
- correct installation sources ([de4c727](https://github.com/cognitivefactory/interactive-clustering/commit/de4c727798f174fbd0c87ae6f6204b09a8d0a131) by Erwan SCHILD).

### Build
- update import information ([fde123e](https://github.com/cognitivefactory/interactive-clustering/commit/fde123e41d3731a8fd3529acf70a40675de13bf6) by Erwan SCHILD).
- fix installation source ([a9d3423](https://github.com/cognitivefactory/interactive-clustering/commit/a9d342375bcdcca5216e3ca1d51848baa5b7d2bb) by Erwan SCHILD).
- change ci configuration ([0d8befd](https://github.com/cognitivefactory/interactive-clustering/commit/0d8befd14ed44f21034fa45a1a450929503b08e2) by Erwan SCHILD).
- init repository ([b248af7](https://github.com/cognitivefactory/interactive-clustering/commit/b248af75b4b8c9fc9f879fa5fe37409baca8ac93) by Erwan SCHILD).

### Code Refactoring
- order import and update documentation ([70e8780](https://github.com/cognitivefactory/interactive-clustering/commit/70e8780749f9a5b8cc0ae261c5f33e4712480dad) by Erwan SCHILD).
- remove local fr_core_news_sm model ([1f9da8f](https://github.com/cognitivefactory/interactive-clustering/commit/1f9da8fee5fc9728114e19753a54a7e4edacc25d) by Erwan SCHILD).
- test fr_core_news_sm installation like a pip package ([b249159](https://github.com/cognitivefactory/interactive-clustering/commit/b2491599b1ff0e606f449fa37208e0a1c31484b4) by Erwan SCHILD).

### Features
- implement Interactive Clustering ([d678d87](https://github.com/cognitivefactory/interactive-clustering/commit/d678d87d0d485cb7b977bb229b9bb8c6fc590c66) by Erwan SCHILD).