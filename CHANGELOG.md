# Changelog

## [0.5.5](https://github.com/sdaza/experiment-utils-pd/compare/v0.5.4...v0.5.5) (2026-04-24)


### Bug Fixes

* update bootstrap sample preparation to include control-group ratio and enhance CI checks ([fa8ed52](https://github.com/sdaza/experiment-utils-pd/commit/fa8ed52435e10a87e7319fb51c51ad1c5ed402f5))
* update bootstrap sample preparation to include control-group ratio and enhance CI checks ([91d19f7](https://github.com/sdaza/experiment-utils-pd/commit/91d19f7bc4b6e2ca7e059ff0e4e0ba55f636f85d))

## [0.5.4](https://github.com/sdaza/experiment-utils-pd/compare/v0.5.3...v0.5.4) (2026-04-17)


### Bug Fixes

* update effect plot display logic and enhance meta-analysis grouping ([2dd3ca4](https://github.com/sdaza/experiment-utils-pd/commit/2dd3ca4ab4b4cb2330649a63e010e81fc8835f5d))
* update effect plot display logic and enhance meta-analysis grouping ([813e0cb](https://github.com/sdaza/experiment-utils-pd/commit/813e0cb1a701f1939d629114d1b4850b08c3c96f))
* update plot effects function calls to use consistent string formatting ([8358783](https://github.com/sdaza/experiment-utils-pd/commit/8358783459d551836c0f4094f6aa3a829995cd13))

## [0.5.3](https://github.com/sdaza/experiment-utils-pd/compare/v0.5.2...v0.5.3) (2026-04-17)


### Bug Fixes

* add effect probabilities and ROPE calculations to ExperimentAnalyzer and update README ([aac5c4d](https://github.com/sdaza/experiment-utils-pd/commit/aac5c4d7866f887f60507a398896d3c2a6a8a02e))
* add probability and ROPE calculations to BootstrapMixin and ExperimentAnalyzer ([b8414b1](https://github.com/sdaza/experiment-utils-pd/commit/b8414b1bc98e90b3dc4a037a56de60634ce8fb6e))
* equivalence plots with several outcomes ([467076a](https://github.com/sdaza/experiment-utils-pd/commit/467076a4285b4259fb6b9f28b478b20f74327bd5))
* equivalence plots with several outcomes ([9216c81](https://github.com/sdaza/experiment-utils-pd/commit/9216c811a529550845f0fc705324646a2a7b7e53))

## [0.5.2](https://github.com/sdaza/experiment-utils-pd/compare/v0.5.1...v0.5.2) (2026-04-12)


### Bug Fixes

* use unadjusted NHST significance in equivalence conclusion logic to match unadjusted TOST p-values, fixing inconsistent labels when MCP corrections were applied

## [0.5.1](https://github.com/sdaza/experiment-utils-pd/compare/v0.5.0...v0.5.1) (2026-04-12)


### Documentation

* update README with TOST equivalence testing documentation ([ebb727b](https://github.com/sdaza/experiment-utils-pd/commit/ebb727beb6b609860c219cf141588b214e7e4bd1))

## [0.5.0](https://github.com/sdaza/experiment-utils-pd/compare/v0.4.0...v0.5.0) (2026-04-01)


### Features

* add plot_equivalence() function and wrapper in ExperimentAnalyzer; update plotting colors and add power_tost() for TOST analysis; bump version to 0.4.0 ([2b7ef84](https://github.com/sdaza/experiment-utils-pd/commit/2b7ef8404ea2810eacaf5682e64adc0d04f76221))
* add plot_equivalence(), class wrapper, and power_tost() ([909bb67](https://github.com/sdaza/experiment-utils-pd/commit/909bb67242e98a8e3188c535aba25e4d0097ba43))
* add pooled_sd column to get_effects() results ([a472efa](https://github.com/sdaza/experiment-utils-pd/commit/a472efad5d15bb0815d8de7240b761c9b202cc4c))
* add test_equivalence() replacing test_non_inferiority() ([ac47e0e](https://github.com/sdaza/experiment-utils-pd/commit/ac47e0e7ce2a37ca23c50124907dbc55396ac780))


### Documentation

* add plot_equivalence() example images ([6bf2c2a](https://github.com/sdaza/experiment-utils-pd/commit/6bf2c2a6b7433bf0a0dca570e2475b2893fe7f8d))
* add TOST equivalence testing design spec ([8756bcb](https://github.com/sdaza/experiment-utils-pd/commit/8756bcbd82f70fe997bc2e88c9c365f6ff10a6db))
* add TOST equivalence testing implementation plan ([a9d2b20](https://github.com/sdaza/experiment-utils-pd/commit/a9d2b20a6fa911371ba2d998afa6fbdf830aff5e))

## [0.4.0](https://github.com/sdaza/experiment-utils-pd/compare/v0.3.5...v0.4.0) (2026-03-30)


### Features

* add df_resid to OLS estimator outputs for t-distribution CI computation ([6e7d8a4](https://github.com/sdaza/experiment-utils-pd/commit/6e7d8a46049c02bce9625b636fbf56a850a10b1d))


### Bug Fixes

* add df_resid parameter for Fieller CI computation in estimators and experiment analyzer ([074ca9f](https://github.com/sdaza/experiment-utils-pd/commit/074ca9fd901508ae7f07da6d521a490b65392d2a))
* enhance color handling and legend display in plotting functions ([54d413c](https://github.com/sdaza/experiment-utils-pd/commit/54d413c12c894864c5848f1a4fbaf712a5eb884a))
* remove unused import and clean up plotting function parameters ([cc55221](https://github.com/sdaza/experiment-utils-pd/commit/cc552213ba1334a8e50cf1c40c9e84a53144fc08))
* use t-distribution for OLS asymptotic CIs to match p-values ([c5c4727](https://github.com/sdaza/experiment-utils-pd/commit/c5c4727ef9b9cf81f5c67adcfe99efc8dd5c04d8))
* use t-distribution in Fieller's method for relative effect CIs ([03037e1](https://github.com/sdaza/experiment-utils-pd/commit/03037e17fc479f74a484710224657a900a2b1319))
* use t-distribution in MCP-adjusted CIs when df_resid available ([9501891](https://github.com/sdaza/experiment-utils-pd/commit/95018914efc504980106d91b7d82e9d8e4fe9aec))
* use t-distribution in plotting fallback and MCP CI computations ([5851ab9](https://github.com/sdaza/experiment-utils-pd/commit/5851ab9bdfdc8a049f3e9cf6dec3f6489c824f6e))

## [0.3.5](https://github.com/sdaza/experiment-utils-pd/compare/v0.3.4...v0.3.5) (2026-03-16)


### Bug Fixes

* update meta-analysis diagnostics handling and improve plotting functions for unique outcomes ([3c4dbe6](https://github.com/sdaza/experiment-utils-pd/commit/3c4dbe6470ce96122367c4052e4c20334a6c19e0))
* update meta-analysis diagnostics handling and improve plotting functions for unique outcomes ([298f47d](https://github.com/sdaza/experiment-utils-pd/commit/298f47d84e72ac0c51b8ac6a172a2c4aa0f06d46))

## [0.3.4](https://github.com/sdaza/experiment-utils-pd/compare/v0.3.3...v0.3.4) (2026-03-14)


### Bug Fixes

* update README with enhanced meta-analysis details and effect visualization improvements ([d79ab4d](https://github.com/sdaza/experiment-utils-pd/commit/d79ab4dbdbc3cd49cdbb175c0ba088a02e4e885d))
* update README with enhanced meta-analysis details and effect visualization improvements ([facdec5](https://github.com/sdaza/experiment-utils-pd/commit/facdec504f05b725e111097903737dd5bc082475))

## [0.3.3](https://github.com/sdaza/experiment-utils-pd/compare/v0.3.2...v0.3.3) (2026-03-13)


### Bug Fixes

* add environment and permissions for PyPI publishing job ([ea7c13b](https://github.com/sdaza/experiment-utils-pd/commit/ea7c13b4aca0723c91762de1d49cb0863522e2af))
* add environment and permissions for PyPI publishing job ([12e674f](https://github.com/sdaza/experiment-utils-pd/commit/12e674f5f6531e2e1fcd2edda5eb54911c700e23))

## [0.3.2](https://github.com/sdaza/experiment-utils-pd/compare/v0.3.1...v0.3.2) (2026-03-13)


### Bug Fixes

* update workflow triggers for manual publishing to PyPI ([8aad234](https://github.com/sdaza/experiment-utils-pd/commit/8aad2340d47f64ae10ed43c401343b2c2f5d302b))
* update workflow triggers for manual publishing to PyPI ([a016a5f](https://github.com/sdaza/experiment-utils-pd/commit/a016a5fff2231075185251f32c37d12a7f235394))

## [0.3.1](https://github.com/sdaza/experiment-utils-pd/compare/v0.3.0...v0.3.1) (2026-03-13)


### Bug Fixes

* add combined label option to plot effects for enhanced clarity ([0a30564](https://github.com/sdaza/experiment-utils-pd/commit/0a30564f35695a66f858f78bee4aba130ee803ac))
* add function to compute pooled estimates from visible plot data ([326ba48](https://github.com/sdaza/experiment-utils-pd/commit/326ba48f32f3372da22458b81b6c2435e2146d1b))
* add function to fill missing panel rows for consistent y-positions in plots ([04cf800](https://github.com/sdaza/experiment-utils-pd/commit/04cf800d26d190413d441e5d839e0ea45d438ffd))
* add missing plt.show() calls to ensure all plots are displayed ([4838e07](https://github.com/sdaza/experiment-utils-pd/commit/4838e077bbf327fc68d1721dea404e1ba5d0b5b5))
* add relative_cap parameter to formatting and plotting functions for enhanced effect label control ([e974385](https://github.com/sdaza/experiment-utils-pd/commit/e974385930a4a0e69ce25d0b95c3825a4d67b82c))
* add support for percentage points display in effect labels and plots ([10e4a58](https://github.com/sdaza/experiment-utils-pd/commit/10e4a58921b704200c7d8422bf05ac40a5b4d826))
* enhance effect visualization with percentage points and combined labels ([e9b5bf6](https://github.com/sdaza/experiment-utils-pd/commit/e9b5bf6fd4fbd7391b795db386d861857939307a))
* enhance plot rendering by adding group metadata support ([0c7fcbb](https://github.com/sdaza/experiment-utils-pd/commit/0c7fcbb83db308d193ce01f552822d1d54f10655))
* handle degenerate standard errors in estimates and plotting for consistency ([35994ec](https://github.com/sdaza/experiment-utils-pd/commit/35994ecd5110db36b1d59dc60935dca28e5a9c13))
* improve handling of confidence intervals and row ordering in panel plots ([eecdb70](https://github.com/sdaza/experiment-utils-pd/commit/eecdb70529eec6a4e18f095bb4148655f48c6235))
* update effect label formatting for values exceeding 100% in _fmt_label and _draw_panels_into_axes ([be2e905](https://github.com/sdaza/experiment-utils-pd/commit/be2e905980d37a45de1d913f7465d5174454a016))
* update package version to 0.3.0 and improve NaN handling in plotting functions ([e5017c9](https://github.com/sdaza/experiment-utils-pd/commit/e5017c9664587bcb3cae70a6af34fa3184306ff8))
* update package version to 0.3.0 and improve NaN handling in plotting functions ([b8943a4](https://github.com/sdaza/experiment-utils-pd/commit/b8943a4dc40b2e94bf21fc14c80924f2adf40d69))

## [0.3.0](https://github.com/sdaza/experiment-utils-pd/compare/v0.2.1...v0.3.0) (2026-03-13)


### Features

* enhance bootstrap functionality with standard error and p-value methods ([c470ca7](https://github.com/sdaza/experiment-utils-pd/commit/c470ca78bd6e5e946859f5176709d7bec91fa3a7))
* enhance bootstrap functionality with standard error and p-value methods ([9eca547](https://github.com/sdaza/experiment-utils-pd/commit/9eca54746391ff07b5f6f93a8a7ac7f8e551875a))

## [0.2.1](https://github.com/sdaza/experiment-utils-pd/compare/v0.2.0...v0.2.1) (2026-03-05)


### Bug Fixes

* correct .gitignore entry for papers directory ([b6a588a](https://github.com/sdaza/experiment-utils-pd/commit/b6a588a86b7039ae7bc1e2f8971dbff447258a67))


### Documentation

* update plot_effects function to include save_path parameter for automatic file saving ([7d0c419](https://github.com/sdaza/experiment-utils-pd/commit/7d0c419ce167f2a49b03d675f46334e2b0cb7567))

## [0.2.0](https://github.com/sdaza/experiment-utils-pd/compare/v0.1.13...v0.2.0) (2026-03-03)


### Features

* Update GitHub Actions workflows for publishing and release management ([cd80782](https://github.com/sdaza/experiment-utils-pd/commit/cd807826618c23e815672a9686fbf860c5b799e3))
