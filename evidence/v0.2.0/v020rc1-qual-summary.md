{
  "build_id": "v0.2.0-rc.1-qual-450k",
  "decode": {
    "c1": {
      "client_latency_ms": {
        "inter_token_latency": {
          "avg_per_run": {
            "count": 5,
            "max": 5.420564955311355,
            "mean": 4.277677813772893,
            "median": 4.094526578998779,
            "min": 3.798276946275946,
            "sample_cv": 0.1559669178632423,
            "sample_stddev": 0.6671762242261307
          },
          "p50_per_run": {
            "count": 5,
            "max": 5.420564955311355,
            "mean": 4.277677813772893,
            "median": 4.094526578998779,
            "min": 3.798276946275946,
            "sample_cv": 0.1559669178632423,
            "sample_stddev": 0.6671762242261307
          },
          "p90_per_run": {
            "count": 5,
            "max": 5.420564955311355,
            "mean": 4.277677813772893,
            "median": 4.094526578998779,
            "min": 3.798276946275946,
            "sample_cv": 0.1559669178632423,
            "sample_stddev": 0.6671762242261307
          },
          "p99_per_run": {
            "count": 5,
            "max": 5.420564955311355,
            "mean": 4.277677813772893,
            "median": 4.094526578998779,
            "min": 3.798276946275946,
            "sample_cv": 0.1559669178632423,
            "sample_stddev": 0.6671762242261307
          }
        },
        "request_latency": {
          "avg_per_run": {
            "count": 5,
            "max": 22557.650739999997,
            "mean": 17870.7029418,
            "median": 17137.157046,
            "min": 15898.78718,
            "sample_cv": 0.15312145169400806,
            "sample_stddev": 2736.3879772407963
          },
          "p50_per_run": {
            "count": 5,
            "max": 22557.650739999997,
            "mean": 17870.7029418,
            "median": 17137.157046,
            "min": 15898.78718,
            "sample_cv": 0.15312145169400806,
            "sample_stddev": 2736.3879772407963
          },
          "p90_per_run": {
            "count": 5,
            "max": 22557.650739999997,
            "mean": 17870.7029418,
            "median": 17137.157046,
            "min": 15898.78718,
            "sample_cv": 0.15312145169400806,
            "sample_stddev": 2736.3879772407963
          },
          "p99_per_run": {
            "count": 5,
            "max": 22557.650739999997,
            "mean": 17870.7029418,
            "median": 17137.157046,
            "min": 15898.78718,
            "sample_cv": 0.15312145169400806,
            "sample_stddev": 2736.3879772407963
          }
        },
        "time_to_first_token": {
          "avg_per_run": {
            "count": 5,
            "max": 370.070705,
            "mean": 353.6122944,
            "median": 348.756707,
            "min": 343.95372699999996,
            "sample_cv": 0.031970381782909695,
            "sample_stddev": 11.30512005509866
          },
          "p50_per_run": {
            "count": 5,
            "max": 370.070705,
            "mean": 353.6122944,
            "median": 348.756707,
            "min": 343.95372699999996,
            "sample_cv": 0.031970381782909695,
            "sample_stddev": 11.30512005509866
          },
          "p90_per_run": {
            "count": 5,
            "max": 370.070705,
            "mean": 353.6122944,
            "median": 348.756707,
            "min": 343.95372699999996,
            "sample_cv": 0.031970381782909695,
            "sample_stddev": 11.30512005509866
          },
          "p99_per_run": {
            "count": 5,
            "max": 370.070705,
            "mean": 353.6122944,
            "median": 348.756707,
            "min": 343.95372699999996,
            "sample_cv": 0.031970381782909695,
            "sample_stddev": 11.30512005509866
          }
        }
      },
      "engine_forward_passes_per_second": {
        "count": 5,
        "max": 66.01110351018801,
        "mean": 54.581447782459485,
        "median": 52.30933123150408,
        "min": 50.55149225894412,
        "sample_cv": 0.11785975450587428,
        "sample_stddev": 6.432956036215871
      },
      "output_tokens_per_forward_per_request": {
        "count": 5,
        "max": 5.862288135593221,
        "mean": 4.875581484600443,
        "median": 5.181657848324515,
        "min": 2.833816425120773,
        "sample_cv": 0.241127206680415,
        "sample_stddev": 1.1756353443244556
      },
      "repetitions": [
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 4.094526578998779,
              "p50": 4.094526578998779,
              "p90": 4.094526578998779,
              "p99": 4.094526578998779
            },
            "request_latency": {
              "avg": 17137.157046,
              "p50": 17137.157046,
              "p90": 17137.157046,
              "p99": 17137.157046
            },
            "time_to_first_token": {
              "avg": 370.070705,
              "p50": 370.070705,
              "p90": 370.070705,
              "p99": 370.070705
            }
          },
          "forward_passes_per_second": 52.43331879872854,
          "id": "r01",
          "output_tokens_per_forward_per_request": 5.181657848324515,
          "speculative": {
            "accept_length": {
              "max": 6.0,
              "mean": 5.017424242424242,
              "median": 6.0,
              "min": 2.825
            },
            "accept_rate": {
              "max": 1.0,
              "mean": 0.8515656565656565,
              "median": 1.0,
              "min": 0.53
            }
          },
          "synthetic_decode_tokens_per_second": 283.09261908596017
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 4.256113812942613,
              "p50": 4.256113812942613,
              "p90": 4.256113812942613,
              "p99": 4.256113812942613
            },
            "request_latency": {
              "avg": 17772.739791,
              "p50": 17772.739791,
              "p90": 17772.739791,
              "p99": 17772.739791
            },
            "time_to_first_token": {
              "avg": 343.95372699999996,
              "p50": 343.95372699999996,
              "p90": 343.95372699999996,
              "p99": 343.95372699999996
            }
          },
          "forward_passes_per_second": 51.60199311293269,
          "id": "r02",
          "output_tokens_per_forward_per_request": 5.3272394881170015,
          "speculative": {
            "accept_length": {
              "max": 6.0,
              "mean": 5.165625,
              "median": 5.825,
              "min": 2.5
            },
            "accept_rate": {
              "max": 1.0,
              "mean": 0.8661458333333333,
              "median": 0.965,
              "min": 0.5
            }
          },
          "synthetic_decode_tokens_per_second": 290.42116842415174
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 3.798276946275946,
              "p50": 3.798276946275946,
              "p90": 3.798276946275946,
              "p99": 3.798276946275946
            },
            "request_latency": {
              "avg": 15898.78718,
              "p50": 15898.78718,
              "p90": 15898.78718,
              "p99": 15898.78718
            },
            "time_to_first_token": {
              "avg": 344.843085,
              "p50": 344.843085,
              "p90": 344.843085,
              "p99": 344.843085
            }
          },
          "forward_passes_per_second": 50.55149225894412,
          "id": "r03",
          "output_tokens_per_forward_per_request": 5.862288135593221,
          "speculative": {
            "accept_length": {
              "max": 5.975,
              "mean": 5.849137931034483,
              "median": 5.85,
              "min": 5.625
            },
            "accept_rate": {
              "max": 0.995,
              "mean": 0.9698275862068966,
              "median": 0.97,
              "min": 0.925
            }
          },
          "synthetic_decode_tokens_per_second": 296.6137952966092
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 5.420564955311355,
              "p50": 5.420564955311355,
              "p90": 5.420564955311355,
              "p99": 5.420564955311355
            },
            "request_latency": {
              "avg": 22557.650739999997,
              "p50": 22557.650739999997,
              "p90": 22557.650739999997,
              "p99": 22557.650739999997
            },
            "time_to_first_token": {
              "avg": 360.437248,
              "p50": 360.437248,
              "p90": 360.437248,
              "p99": 360.437248
            }
          },
          "forward_passes_per_second": 66.01110351018801,
          "id": "r04",
          "output_tokens_per_forward_per_request": 2.833816425120773,
          "speculative": {
            "accept_length": {
              "max": 5.925,
              "mean": 2.9244897959183676,
              "median": 2.55,
              "min": 2.175
            },
            "accept_rate": {
              "max": 0.985,
              "mean": 0.549047619047619,
              "median": 0.5166666666666667,
              "min": 0.35
            }
          },
          "synthetic_decode_tokens_per_second": 174.23305903641108
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 3.818906775335775,
              "p50": 3.818906775335775,
              "p90": 3.818906775335775,
              "p99": 3.818906775335775
            },
            "request_latency": {
              "avg": 15987.179951999999,
              "p50": 15987.179951999999,
              "p90": 15987.179951999999,
              "p99": 15987.179951999999
            },
            "time_to_first_token": {
              "avg": 348.756707,
              "p50": 348.756707,
              "p90": 348.756707,
              "p99": 348.756707
            }
          },
          "forward_passes_per_second": 52.30933123150408,
          "id": "r05",
          "output_tokens_per_forward_per_request": 5.172905525846702,
          "speculative": {
            "accept_length": {
              "max": 5.975,
              "mean": 5.267424242424242,
              "median": 5.5,
              "min": 3.5
            },
            "accept_rate": {
              "max": 0.995,
              "mean": 0.8736868686868687,
              "median": 0.9,
              "min": 0.62
            }
          },
          "synthetic_decode_tokens_per_second": 276.2220415273493
        }
      ],
      "speculative": {
        "accept_length": {
          "max_per_run": {
            "count": 5,
            "max": 6.0,
            "mean": 5.975,
            "median": 5.975,
            "min": 5.925,
            "sample_cv": 0.005124455528835113,
            "sample_stddev": 0.030618621784789798
          },
          "mean_per_run": {
            "count": 5,
            "max": 5.849137931034483,
            "mean": 4.844820242360266,
            "median": 5.165625,
            "min": 2.9244897959183676,
            "sample_cv": 0.2309448257905403,
            "sample_stddev": 1.118886166858375
          },
          "median_per_run": {
            "count": 5,
            "max": 6.0,
            "mean": 5.1450000000000005,
            "median": 5.825,
            "min": 2.55,
            "sample_cv": 0.2841714091728629,
            "sample_stddev": 1.4620619001943798
          },
          "min_per_run": {
            "count": 5,
            "max": 5.625,
            "mean": 3.325,
            "median": 2.825,
            "min": 2.175,
            "sample_cv": 0.4138413074292493,
            "sample_stddev": 1.376022347202254
          }
        },
        "accept_rate": {
          "max_per_run": {
            "count": 5,
            "max": 1.0,
            "mean": 0.9949999999999999,
            "median": 0.995,
            "min": 0.985,
            "sample_cv": 0.00615449684116377,
            "sample_stddev": 0.006123724356957951
          },
          "mean_per_run": {
            "count": 5,
            "max": 0.9698275862068966,
            "mean": 0.8220547127680747,
            "median": 0.8661458333333333,
            "min": 0.549047619047619,
            "sample_cv": 0.1941113179959779,
            "sample_stddev": 0.159570123760216
          },
          "median_per_run": {
            "count": 5,
            "max": 1.0,
            "mean": 0.8703333333333333,
            "median": 0.965,
            "min": 0.5166666666666667,
            "sample_cv": 0.2309923589467093,
            "sample_stddev": 0.20104034973661933
          },
          "min_per_run": {
            "count": 5,
            "max": 0.925,
            "mean": 0.585,
            "median": 0.53,
            "min": 0.35,
            "sample_cv": 0.36492824000357604,
            "sample_stddev": 0.21348302040209197
          }
        }
      },
      "synthetic_decode_tokens_per_second": {
        "count": 5,
        "max": 296.6137952966092,
        "mean": 264.11653667409627,
        "median": 283.09261908596017,
        "min": 174.23305903641108,
        "sample_cv": 0.19244290953450763,
        "sample_stddev": 50.82735477374057
      }
    },
    "c2": {
      "client_latency_ms": {
        "inter_token_latency": {
          "avg_per_run": {
            "count": 5,
            "max": 7.219691782417582,
            "mean": 6.518258151746032,
            "median": 6.507071994505495,
            "min": 5.689596457264956,
            "sample_cv": 0.09372575616696581,
            "sample_stddev": 0.6109286741638859
          },
          "p50_per_run": {
            "count": 5,
            "max": 7.219691782417582,
            "mean": 6.518258151746032,
            "median": 6.507071994505495,
            "min": 5.689596457264956,
            "sample_cv": 0.09372575616696581,
            "sample_stddev": 0.6109286741638859
          },
          "p90_per_run": {
            "count": 5,
            "max": 7.7424778500854705,
            "mean": 7.16291807793895,
            "median": 7.0766576000732595,
            "min": 6.43473511015873,
            "sample_cv": 0.07037423747681804,
            "sample_stddev": 0.5040848978438687
          },
          "p99_per_run": {
            "count": 5,
            "max": 7.91455641841514,
            "mean": 7.307966561332355,
            "median": 7.274525140007326,
            "min": 6.602391307059828,
            "sample_cv": 0.06728585794004335,
            "sample_stddev": 0.491722799876396
          }
        },
        "request_latency": {
          "avg_per_run": {
            "count": 5,
            "max": 30158.9509845,
            "mean": 27275.606645,
            "median": 27219.326488,
            "min": 23883.765799,
            "sample_cv": 0.09192034079551498,
            "sample_stddev": 2507.1830582128127
          },
          "p50_per_run": {
            "count": 5,
            "max": 30158.9509845,
            "mean": 27275.606645,
            "median": 27219.326488,
            "min": 23883.765799,
            "sample_cv": 0.09192034079551498,
            "sample_stddev": 2507.1830582128127
          },
          "p90_per_run": {
            "count": 5,
            "max": 32298.1811179,
            "mean": 29913.68952244,
            "median": 29553.728005599998,
            "min": 26923.2319174,
            "sample_cv": 0.06930624325234869,
            "sample_stddev": 2073.205442617461
          },
          "p99_per_run": {
            "count": 5,
            "max": 33003.21602239,
            "mean": 30507.258169864002,
            "median": 30364.274914359998,
            "min": 27607.11179404,
            "sample_cv": 0.06632262204393012,
            "sample_stddev": 2023.3213531964896
          }
        },
        "time_to_first_token": {
          "avg_per_run": {
            "count": 5,
            "max": 594.3131354999999,
            "mean": 583.3395135999999,
            "median": 584.8683065,
            "min": 572.8666704999999,
            "sample_cv": 0.01687618209061867,
            "sample_stddev": 9.844543852166524
          },
          "p50_per_run": {
            "count": 5,
            "max": 594.3131354999999,
            "mean": 583.3395135999999,
            "median": 584.8683065,
            "min": 572.8666704999999,
            "sample_cv": 0.01687618209061867,
            "sample_stddev": 9.844543852166524
          },
          "p90_per_run": {
            "count": 5,
            "max": 596.7449717000001,
            "mean": 586.98524128,
            "median": 592.7343218,
            "min": 574.5821893,
            "sample_cv": 0.01928205333226643,
            "sample_stddev": 11.318280727614237
          },
          "p99_per_run": {
            "count": 5,
            "max": 599.41722137,
            "mean": 587.8055300079999,
            "median": 593.10748898,
            "min": 574.96818103,
            "sample_cv": 0.020198811251672746,
            "sample_stddev": 11.87297295332105
          }
        }
      },
      "engine_forward_passes_per_second": {
        "count": 5,
        "max": 42.11401746335851,
        "mean": 39.13475840053499,
        "median": 38.05023402293512,
        "min": 36.94296944531069,
        "sample_cv": 0.06099002840595715,
        "sample_stddev": 2.3868300265088993
      },
      "output_tokens_per_forward_per_request": {
        "count": 5,
        "max": 4.574257425742574,
        "mean": 3.919203784276166,
        "median": 4.042536115569823,
        "min": 3.262784090909091,
        "sample_cv": 0.15168876392548827,
        "sample_stddev": 0.5944991776089475
      },
      "repetitions": [
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 6.507071994505495,
              "p50": 6.507071994505495,
              "p90": 7.040317905030524,
              "p99": 7.160298234898656
            },
            "request_latency": {
              "avg": 27219.326488,
              "p50": 27219.326488,
              "p90": 29404.6840104,
              "p99": 29896.38945294
            },
            "time_to_first_token": {
              "avg": 572.8666704999999,
              "p50": 572.8666704999999,
              "p90": 574.5821893,
              "p99": 574.96818103
            }
          },
          "forward_passes_per_second": 38.05023402293512,
          "id": "r01",
          "output_tokens_per_forward_per_request": 4.042536115569823,
          "speculative": {
            "accept_length": {
              "max": 4.8625,
              "mean": 4.02325,
              "median": 4.15,
              "min": 3.075
            },
            "accept_rate": {
              "max": 0.9666666666666667,
              "mean": 0.65975,
              "median": 0.6725,
              "min": 0.415
            }
          },
          "synthetic_decode_tokens_per_second": 306.91284344611284
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 7.219691782417582,
              "p50": 7.219691782417582,
              "p90": 7.520401924346763,
              "p99": 7.588061706280829
            },
            "request_latency": {
              "avg": 30158.9509845,
              "p50": 30158.9509845,
              "p90": 31388.6225609,
              "p99": 31665.29866559
            },
            "time_to_first_token": {
              "avg": 594.3131354999999,
              "p50": 594.3131354999999,
              "p90": 596.0495903,
              "p99": 596.4402926299999
            }
          },
          "forward_passes_per_second": 41.26918497519074,
          "id": "r02",
          "output_tokens_per_forward_per_request": 3.342911877394636,
          "speculative": {
            "accept_length": {
              "max": 5.4,
              "mean": 3.35323275862069,
              "median": 3.1125,
              "min": 2.0875
            },
            "accept_rate": {
              "max": 0.9833333333333333,
              "mean": 0.6059339080459769,
              "median": 0.6075,
              "min": 0.3625
            }
          },
          "synthetic_decode_tokens_per_second": 271.5985778385013
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 5.689596457264956,
              "p50": 5.689596457264956,
              "p90": 6.43473511015873,
              "p99": 6.602391307059828
            },
            "request_latency": {
              "avg": 23883.765799,
              "p50": 23883.765799,
              "p90": 26923.2319174,
              "p99": 27607.11179404
            },
            "time_to_first_token": {
              "avg": 584.8683065,
              "p50": 584.8683065,
              "p90": 596.7449717000001,
              "p99": 599.41722137
            }
          },
          "forward_passes_per_second": 37.29738609587991,
          "id": "r03",
          "output_tokens_per_forward_per_request": 4.574257425742574,
          "speculative": {
            "accept_length": {
              "max": 5.2625,
              "mean": 4.732926829268293,
              "median": 4.65,
              "min": 4.3125
            },
            "accept_rate": {
              "max": 0.8525,
              "mean": 0.7465853658536585,
              "median": 0.73,
              "min": 0.6625
            }
          },
          "synthetic_decode_tokens_per_second": 348.14313735601604
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 6.977684213064713,
              "p50": 6.977684213064713,
              "p90": 7.7424778500854705,
              "p99": 7.91455641841514
            },
            "request_latency": {
              "avg": 29164.6926535,
              "p50": 29164.6926535,
              "p90": 32298.1811179,
              "p99": 33003.21602239
            },
            "time_to_first_token": {
              "avg": 591.075801,
              "p50": 591.075801,
              "p90": 592.7343218,
              "p99": 593.10748898
            }
          },
          "forward_passes_per_second": 42.11401746335851,
          "id": "r04",
          "output_tokens_per_forward_per_request": 3.262784090909091,
          "speculative": {
            "accept_length": {
              "max": 4.3375,
              "mean": 3.3379901960784317,
              "median": 3.2875,
              "min": 2.7125
            },
            "accept_rate": {
              "max": 0.8458333333333333,
              "mean": 0.6595261437908496,
              "median": 0.6458333333333334,
              "min": 0.4875
            }
          },
          "synthetic_decode_tokens_per_second": 278.92495730086245
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 6.197246311477412,
              "p50": 6.197246311477412,
              "p90": 7.0766576000732595,
              "p99": 7.274525140007326
            },
            "request_latency": {
              "avg": 25951.2973,
              "p50": 25951.2973,
              "p90": 29553.728005599998,
              "p99": 30364.274914359998
            },
            "time_to_first_token": {
              "avg": 573.5736545,
              "p50": 573.5736545,
              "p90": 574.8151333,
              "p99": 575.0944660299999
            }
          },
          "forward_passes_per_second": 36.94296944531069,
          "id": "r05",
          "output_tokens_per_forward_per_request": 4.373529411764705,
          "speculative": {
            "accept_length": {
              "max": 4.6875,
              "mean": 4.466369047619048,
              "median": 4.5375,
              "min": 4.1
            },
            "accept_rate": {
              "max": 0.7375,
              "mean": 0.6932738095238096,
              "median": 0.7075,
              "min": 0.62
            }
          },
          "synthetic_decode_tokens_per_second": 329.03652807919264
        }
      ],
      "speculative": {
        "accept_length": {
          "max_per_run": {
            "count": 5,
            "max": 5.4,
            "mean": 4.91,
            "median": 4.8625,
            "min": 4.3375,
            "sample_cv": 0.08782911241015412,
            "sample_stddev": 0.43124094193385676
          },
          "mean_per_run": {
            "count": 5,
            "max": 4.732926829268293,
            "mean": 3.9827537663172925,
            "median": 4.02325,
            "min": 3.3379901960784317,
            "sample_cv": 0.15930886859209903,
            "sample_stddev": 0.6344879963929291
          },
          "median_per_run": {
            "count": 5,
            "max": 4.65,
            "mean": 3.9475000000000002,
            "median": 4.15,
            "min": 3.1125,
            "sample_cv": 0.17981773921423083,
            "sample_stddev": 0.7098305255481763
          },
          "min_per_run": {
            "count": 5,
            "max": 4.3125,
            "mean": 3.2575000000000003,
            "median": 3.075,
            "min": 2.0875,
            "sample_cv": 0.2880607646252469,
            "sample_stddev": 0.9383579407667417
          }
        },
        "accept_rate": {
          "max_per_run": {
            "count": 5,
            "max": 0.9833333333333333,
            "mean": 0.8771666666666667,
            "median": 0.8525,
            "min": 0.7375,
            "sample_cv": 0.11454168641183758,
            "sample_stddev": 0.1004721492642502
          },
          "mean_per_run": {
            "count": 5,
            "max": 0.7465853658536585,
            "mean": 0.6730138454428589,
            "median": 0.65975,
            "min": 0.6059339080459769,
            "sample_cv": 0.07677996436953263,
            "sample_stddev": 0.05167397907330485
          },
          "median_per_run": {
            "count": 5,
            "max": 0.73,
            "mean": 0.6726666666666666,
            "median": 0.6725,
            "min": 0.6075,
            "sample_cv": 0.07232382436069047,
            "sample_stddev": 0.048649825853291126
          },
          "min_per_run": {
            "count": 5,
            "max": 0.6625,
            "mean": 0.5095,
            "median": 0.4875,
            "min": 0.3625,
            "sample_cv": 0.2533386525579069,
            "sample_stddev": 0.12907604347825355
          }
        }
      },
      "synthetic_decode_tokens_per_second": {
        "count": 5,
        "max": 348.14313735601604,
        "mean": 306.92320880413706,
        "median": 306.91284344611284,
        "min": 271.5985778385013,
        "sample_cv": 0.10582466071672335,
        "sample_stddev": 32.48004443778584
      }
    },
    "c3": {
      "client_latency_ms": {
        "inter_token_latency": {
          "avg_per_run": {
            "count": 5,
            "max": 8.467886715750916,
            "mean": 7.450394691070412,
            "median": 7.416096424175824,
            "min": 6.344840526088725,
            "sample_cv": 0.12272268576416265,
            "sample_stddev": 0.9143324464912198
          },
          "p50_per_run": {
            "count": 5,
            "max": 8.540918873748474,
            "mean": 7.669930581489621,
            "median": 8.082484796581197,
            "min": 6.5276812017094015,
            "sample_cv": 0.12717455294839186,
            "sample_stddev": 0.9754199928461418
          },
          "p90_per_run": {
            "count": 5,
            "max": 9.07524091086691,
            "mean": 7.937929553113553,
            "median": 8.119023448986567,
            "min": 6.710492203614163,
            "sample_cv": 0.12479753327741,
            "sample_stddev": 0.9906340275584249
          },
          "p99_per_run": {
            "count": 5,
            "max": 9.208474005592185,
            "mean": 7.998229321728937,
            "median": 8.127244645777777,
            "min": 6.7516246790427346,
            "sample_cv": 0.12526132019331354,
            "sample_stddev": 1.0018687640486375
          }
        },
        "request_latency": {
          "avg_per_run": {
            "count": 5,
            "max": 35277.14382133333,
            "mean": 31112.194379066663,
            "median": 30962.72694033333,
            "min": 26568.97338133333,
            "sample_cv": 0.12047853867883618,
            "sample_stddev": 3748.3517138818524
          },
          "p50_per_run": {
            "count": 5,
            "max": 35578.177942999995,
            "mean": 32007.765987599996,
            "median": 33693.492759,
            "min": 27305.712935,
            "sample_cv": 0.12506593230739013,
            "sample_stddev": 4003.081094315965
          },
          "p90_per_run": {
            "count": 5,
            "max": 37775.292696599994,
            "mean": 33105.118052399994,
            "median": 33842.7168982,
            "min": 28057.6102054,
            "sample_cv": 0.12276687652855803,
            "sample_stddev": 4064.2119404023274
          },
          "p99_per_run": {
            "count": 5,
            "max": 38320.23822306,
            "mean": 33352.02226698,
            "median": 33876.29232952,
            "min": 28226.78709124,
            "sample_cv": 0.1232284639680145,
            "sample_stddev": 4109.918474186962
          }
        },
        "time_to_first_token": {
          "avg_per_run": {
            "count": 5,
            "max": 618.0417573333333,
            "mean": 602.8281191333333,
            "median": 601.1477203333334,
            "min": 586.851427,
            "sample_cv": 0.021978040925420224,
            "sample_stddev": 13.248981073306497
          },
          "p50_per_run": {
            "count": 5,
            "max": 615.043373,
            "mean": 600.8479302,
            "median": 602.318446,
            "min": 578.966186,
            "sample_cv": 0.02432675650770066,
            "sample_stddev": 14.616681296131322
          },
          "p90_per_run": {
            "count": 5,
            "max": 629.06905,
            "mean": 608.98213884,
            "median": 602.9558132,
            "min": 595.6171063999999,
            "sample_cv": 0.022181690504181614,
            "sample_stddev": 13.508253326323436
          },
          "p99_per_run": {
            "count": 5,
            "max": 632.7529453,
            "mean": 610.812335784,
            "median": 606.1744111,
            "min": 595.70747594,
            "sample_cv": 0.023454454270739153,
            "sample_stddev": 14.326269997649195
          }
        }
      },
      "engine_forward_passes_per_second": {
        "count": 5,
        "max": 33.56622659558965,
        "mean": 31.081380535448556,
        "median": 30.884028754909544,
        "min": 29.8304157035995,
        "sample_cv": 0.047462597726489024,
        "sample_stddev": 1.4752030611379208
      },
      "output_tokens_per_forward_per_request": {
        "count": 5,
        "max": 5.519561815336463,
        "mean": 4.458351566965074,
        "median": 4.344490934449094,
        "min": 3.5936708860759494,
        "sample_cv": 0.17600886120708406,
        "sample_stddev": 0.7847093821623415
      },
      "repetitions": [
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 7.416096424175824,
              "p50": 8.082484796581197,
              "p90": 8.119023448986567,
              "p99": 8.127244645777777
            },
            "request_latency": {
              "avg": 30962.72694033333,
              "p50": 33693.492759,
              "p90": 33842.7168982,
              "p99": 33876.29232952
            },
            "time_to_first_token": {
              "avg": 593.8120833333334,
              "p50": 595.215464,
              "p90": 595.6171063999999,
              "p99": 595.70747594
            }
          },
          "forward_passes_per_second": 29.8304157035995,
          "id": "r01",
          "output_tokens_per_forward_per_request": 4.344490934449094,
          "speculative": {
            "accept_length": {
              "max": 5.191666666666666,
              "mean": 4.329931972789115,
              "median": 4.341666666666667,
              "min": 3.908333333333333
            },
            "accept_rate": {
              "max": 0.8383333333333334,
              "mean": 0.6659863945578232,
              "median": 0.6683333333333333,
              "min": 0.5816666666666667
            }
          },
          "synthetic_decode_tokens_per_second": 387.12879535961054
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 8.467886715750916,
              "p50": 8.540918873748474,
              "p90": 8.63014843956044,
              "p99": 8.650225091868132
            },
            "request_latency": {
              "avg": 35277.14382133333,
              "p50": 35578.177942999995,
              "p90": 35939.488539,
              "p99": 36020.783423099994
            },
            "time_to_first_token": {
              "avg": 601.1477203333334,
              "p50": 602.318446,
              "p90": 602.9558132,
              "p99": 603.09922082
            }
          },
          "forward_passes_per_second": 33.56622659558965,
          "id": "r02",
          "output_tokens_per_forward_per_request": 3.5936708860759494,
          "speculative": {
            "accept_length": {
              "max": 5.383333333333334,
              "mean": 3.586689814814815,
              "median": 3.191666666666667,
              "min": 2.5083333333333333
            },
            "accept_rate": {
              "max": 0.8766666666666667,
              "mean": 0.6594521604938272,
              "median": 0.6633333333333333,
              "min": 0.445
            }
          },
          "synthetic_decode_tokens_per_second": 349.53229033868746
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 6.344840526088725,
              "p50": 6.5276812017094015,
              "p90": 6.710492203614163,
              "p99": 6.7516246790427346
            },
            "request_latency": {
              "avg": 26568.97338133333,
              "p50": 27305.712935,
              "p90": 28057.6102054,
              "p99": 28226.78709124
            },
            "time_to_first_token": {
              "avg": 586.851427,
              "p50": 578.966186,
              "p90": 601.176982,
              "p99": 606.1744111
            }
          },
          "forward_passes_per_second": 31.004352120101522,
          "id": "r03",
          "output_tokens_per_forward_per_request": 5.519561815336463,
          "speculative": {
            "accept_length": {
              "max": 5.966666666666667,
              "mean": 5.29781746031746,
              "median": 5.758333333333334,
              "min": 3.8583333333333334
            },
            "accept_rate": {
              "max": 0.9933333333333333,
              "mean": 0.859563492063492,
              "median": 0.9516666666666667,
              "min": 0.5716666666666667
            }
          },
          "synthetic_decode_tokens_per_second": 522.0671813049987
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 8.245677357916158,
              "p50": 8.483093823199022,
              "p90": 9.07524091086691,
              "p99": 9.208474005592185
            },
            "request_latency": {
              "avg": 34380.33638833334,
              "p50": 35353.312579,
              "p90": 37775.292696599994,
              "p99": 38320.23822306
            },
            "time_to_first_token": {
              "avg": 614.2876076666666,
              "p50": 615.043373,
              "p90": 616.0917426,
              "p99": 616.3276257599999
            }
          },
          "forward_passes_per_second": 30.884028754909544,
          "id": "r04",
          "output_tokens_per_forward_per_request": 3.881828316610925,
          "speculative": {
            "accept_length": {
              "max": 4.725,
              "mean": 3.868103448275862,
              "median": 4.025,
              "min": 2.8333333333333335
            },
            "accept_rate": {
              "max": 0.8166666666666667,
              "mean": 0.6398659003831417,
              "median": 0.6333333333333333,
              "min": 0.5416666666666666
            }
          },
          "synthetic_decode_tokens_per_second": 372.3271521800697
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 6.777472431420431,
              "p50": 6.715474212210013,
              "p90": 7.154742762539683,
              "p99": 7.253578186363859
            },
            "request_latency": {
              "avg": 28371.791363999997,
              "p50": 28108.133722,
              "p90": 29910.481922799998,
              "p99": 30316.010267979997
            },
            "time_to_first_token": {
              "avg": 618.0417573333333,
              "p50": 612.696182,
              "p90": 629.06905,
              "p99": 632.7529453
            }
          },
          "forward_passes_per_second": 30.12187950304258,
          "id": "r05",
          "output_tokens_per_forward_per_request": 4.952205882352941,
          "speculative": {
            "accept_length": {
              "max": 5.475,
              "mean": 4.8956060606060605,
              "median": 4.9,
              "min": 4.308333333333334
            },
            "accept_rate": {
              "max": 0.895,
              "mean": 0.7791212121212121,
              "median": 0.78,
              "min": 0.6616666666666666
            }
          },
          "synthetic_decode_tokens_per_second": 447.62088934884946
        }
      ],
      "speculative": {
        "accept_length": {
          "max_per_run": {
            "count": 5,
            "max": 5.966666666666667,
            "mean": 5.348333333333334,
            "median": 5.383333333333334,
            "min": 4.725,
            "sample_cv": 0.08427818673591707,
            "sample_stddev": 0.4507478353925965
          },
          "mean_per_run": {
            "count": 5,
            "max": 5.29781746031746,
            "mean": 4.395629751360663,
            "median": 4.329931972789115,
            "min": 3.586689814814815,
            "sample_cv": 0.1609021758045014,
            "sample_stddev": 0.7072663910249302
          },
          "median_per_run": {
            "count": 5,
            "max": 5.758333333333334,
            "mean": 4.443333333333333,
            "median": 4.341666666666667,
            "min": 3.191666666666667,
            "sample_cv": 0.2161529739187856,
            "sample_stddev": 0.9604397141124708
          },
          "min_per_run": {
            "count": 5,
            "max": 4.308333333333334,
            "mean": 3.4833333333333334,
            "median": 3.8583333333333334,
            "min": 2.5083333333333333,
            "sample_cv": 0.2212110288146762,
            "sample_stddev": 0.7705517503711221
          }
        },
        "accept_rate": {
          "max_per_run": {
            "count": 5,
            "max": 0.9933333333333333,
            "mean": 0.884,
            "median": 0.8766666666666667,
            "min": 0.8166666666666667,
            "sample_cv": 0.07744486692737046,
            "sample_stddev": 0.06846126236379549
          },
          "mean_per_run": {
            "count": 5,
            "max": 0.859563492063492,
            "mean": 0.7207978319238992,
            "median": 0.6659863945578232,
            "min": 0.6398659003831417,
            "sample_cv": 0.13157017837479712,
            "sample_stddev": 0.09483549931839445
          },
          "median_per_run": {
            "count": 5,
            "max": 0.9516666666666667,
            "mean": 0.7393333333333334,
            "median": 0.6683333333333333,
            "min": 0.6333333333333333,
            "sample_cv": 0.1773778500697516,
            "sample_stddev": 0.1311413571515697
          },
          "min_per_run": {
            "count": 5,
            "max": 0.6616666666666666,
            "mean": 0.5603333333333333,
            "median": 0.5716666666666667,
            "min": 0.445,
            "sample_cv": 0.1396774451747703,
            "sample_stddev": 0.07826592844626296
          }
        }
      },
      "synthetic_decode_tokens_per_second": {
        "count": 5,
        "max": 522.0671813049987,
        "mean": 415.73526170644317,
        "median": 387.12879535961054,
        "min": 349.53229033868746,
        "sample_cv": 0.16755223343913553,
        "sample_stddev": 69.65737161831807
      }
    }
  },
  "engine": "sglang",
  "interpretation": "same-process engineering regression signal; synthetic fixed-window output rate is not expected production, interactive, or application throughput and includes path-dependent speculative acceptance; repetitions are prompt-path subsamples, not independent deployment replicates",
  "mode": "glm-qualification",
  "prefill": {
    "128k-c1": {
      "median_ttft_ms": 22222.822943,
      "prompt_tokens_per_second": 5893.3675755709855,
      "requests": 5
    },
    "32k-c1": {
      "median_ttft_ms": 5598.561164,
      "prompt_tokens_per_second": 5870.77217651838,
      "requests": 5
    },
    "64k-c1": {
      "median_ttft_ms": 11134.80747,
      "prompt_tokens_per_second": 5889.812804549044,
      "requests": 5
    },
    "8k-c1": {
      "median_ttft_ms": 1560.518983,
      "prompt_tokens_per_second": 5245.1022806439405,
      "requests": 5
    }
  },
  "schema_version": "1.2"
}
