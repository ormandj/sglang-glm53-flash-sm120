{
  "build_id": "v0.2.0-rc.1-cohort-450k",
  "decode": {
    "c4": {
      "client_latency_ms": {
        "inter_token_latency": {
          "avg_per_run": {
            "count": 5,
            "max": 11.581014049267399,
            "mean": 10.983174032283271,
            "median": 10.93502481190476,
            "min": 10.315532854395602,
            "sample_cv": 0.04205142325290183,
            "sample_stddev": 0.4618580998918243
          },
          "p50_per_run": {
            "count": 5,
            "max": 11.598826365689865,
            "mean": 11.040385517875457,
            "median": 11.240345978510378,
            "min": 10.137575257387056,
            "sample_cv": 0.050449819797474774,
            "sample_stddev": 0.556985459871467
          },
          "p90_per_run": {
            "count": 5,
            "max": 12.714694467326007,
            "mean": 11.760260781025641,
            "median": 11.893541637680098,
            "min": 10.873773882661782,
            "sample_cv": 0.06156236977775865,
            "sample_stddev": 0.7239895228843732
          },
          "p99_per_run": {
            "count": 5,
            "max": 13.018814065413919,
            "mean": 11.926962503992673,
            "median": 11.994636594317459,
            "min": 11.128150688705738,
            "sample_cv": 0.06276149652606977,
            "sample_stddev": 0.7485540157609006
          }
        },
        "request_latency": {
          "avg_per_run": {
            "count": 5,
            "max": 54939.2247385,
            "mean": 52530.63121755,
            "median": 52368.444978499996,
            "min": 49803.5027785,
            "sample_cv": 0.03565618269464627,
            "sample_stddev": 1873.0417837580515
          },
          "p50_per_run": {
            "count": 5,
            "max": 57216.349110999996,
            "mean": 53398.53266619999,
            "median": 53685.052888499995,
            "min": 49490.209274,
            "sample_cv": 0.052770521396866825,
            "sample_stddev": 2817.868410622999
          },
          "p90_per_run": {
            "count": 5,
            "max": 57495.03104369999,
            "mean": 54946.09647971999,
            "median": 55049.916422099996,
            "min": 53471.3397854,
            "sample_cv": 0.029300855608267572,
            "sample_stddev": 1609.9676391902146
          },
          "p99_per_run": {
            "count": 5,
            "max": 57540.38674287,
            "mean": 55222.100408052,
            "median": 55158.85013691,
            "min": 53690.519378029996,
            "sample_cv": 0.02645297938856849,
            "sample_stddev": 1460.7890838876592
          }
        },
        "time_to_first_token": {
          "avg_per_run": {
            "count": 5,
            "max": 7589.518373749999,
            "mean": 7554.53355535,
            "median": 7561.39573975,
            "min": 7514.97220675,
            "sample_cv": 0.0037534342928570693,
            "sample_stddev": 28.355445313190128
          },
          "p50_per_run": {
            "count": 5,
            "max": 7690.770584,
            "mean": 7646.223099499999,
            "median": 7642.9162865,
            "min": 7617.029555499999,
            "sample_cv": 0.00363073664356126,
            "sample_stddev": 27.7614223921992
          },
          "p90_per_run": {
            "count": 5,
            "max": 10345.5713514,
            "mean": 10310.380220160001,
            "median": 10327.6748312,
            "min": 10241.4105222,
            "sample_cv": 0.003995462805652573,
            "sample_stddev": 41.19474068178528
          },
          "p99_per_run": {
            "count": 5,
            "max": 10963.93798254,
            "mean": 10910.574237456,
            "median": 10922.62281092,
            "min": 10833.06603552,
            "sample_cv": 0.004415106368840352,
            "sample_stddev": 48.17134580349745
          }
        }
      },
      "engine_forward_passes_per_second": {
        "count": 5,
        "max": 32.00745534072723,
        "mean": 31.333139414189095,
        "median": 31.50894436180303,
        "min": 30.369219127044477,
        "sample_cv": 0.019401054358402847,
        "sample_stddev": 0.6078959409940974
      },
      "output_tokens_per_forward_per_request": {
        "count": 5,
        "max": 3.3156836461126007,
        "mean": 3.0630384728637323,
        "median": 3.067702552719201,
        "min": 2.8286924939467313,
        "sample_cv": 0.05674543523216486,
        "sample_stddev": 0.1738134512755181
      },
      "repetitions": [
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 11.581014049267399,
              "p50": 11.598826365689865,
              "p90": 12.714694467326007,
              "p99": 13.018814065413919
            },
            "request_latency": {
              "avg": 54939.2247385,
              "p50": 57216.349110999996,
              "p90": 57495.03104369999,
              "p99": 57540.38674287
            },
            "time_to_first_token": {
              "avg": 7514.97220675,
              "p50": 7617.029555499999,
              "p90": 10241.4105222,
              "p99": 10833.06603552
            }
          },
          "forward_passes_per_second": 32.00745534072723,
          "id": "r01",
          "output_tokens_per_forward_per_request": 2.8286924939467313,
          "speculative": {
            "accept_length": {
              "max": 3.41875,
              "mean": 2.8467147435897435,
              "median": 2.8125,
              "min": 2.49375
            },
            "accept_rate": {
              "max": 0.675,
              "mean": 0.5990331196581197,
              "median": 0.6020833333333333,
              "min": 0.48375
            }
          },
          "synthetic_decode_tokens_per_second": 366.48113143468447
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 10.315532854395602,
              "p50": 10.137575257387056,
              "p90": 10.873773882661782,
              "p99": 11.128150688705738
            },
            "request_latency": {
              "avg": 49803.5027785,
              "p50": 49490.209274,
              "p90": 53471.3397854,
              "p99": 54336.09585614
            },
            "time_to_first_token": {
              "avg": 7561.39573975,
              "p50": 7649.214613499999,
              "p90": 10305.2463295,
              "p99": 10906.414822749999
            }
          },
          "forward_passes_per_second": 31.50894436180303,
          "id": "r02",
          "output_tokens_per_forward_per_request": 3.3156836461126007,
          "speculative": {
            "accept_length": {
              "max": 4.55,
              "mean": 3.2914383561643836,
              "median": 3.09375,
              "min": 2.64375
            },
            "accept_rate": {
              "max": 0.7729166666666667,
              "mean": 0.6525114155251142,
              "median": 0.645,
              "min": 0.5479166666666667
            }
          },
          "synthetic_decode_tokens_per_second": 409.87935669154285
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 11.185012261843713,
              "p50": 11.29091204041514,
              "p90": 12.084988330598291,
              "p99": 12.159797606026862
            },
            "request_latency": {
              "avg": 53342.2793905,
              "p50": 54251.712375999996,
              "p90": 55049.916422099996,
              "p99": 55158.85013691
            },
            "time_to_first_token": {
              "avg": 7539.65417825,
              "p50": 7631.184458,
              "p90": 10327.6748312,
              "p99": 10922.62281092
            }
          },
          "forward_passes_per_second": 31.55874460185326,
          "id": "r03",
          "output_tokens_per_forward_per_request": 3.0214617169373548,
          "speculative": {
            "accept_length": {
              "max": 3.86875,
              "mean": 3.0210090361445783,
              "median": 2.95625,
              "min": 2.49375
            },
            "accept_rate": {
              "max": 0.7270833333333333,
              "mean": 0.6079267068273093,
              "median": 0.6270833333333333,
              "min": 0.475
            }
          },
          "synthetic_decode_tokens_per_second": 380.5270730753362
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 10.93502481190476,
              "p50": 11.240345978510378,
              "p90": 11.893541637680098,
              "p99": 11.994636594317459
            },
            "request_latency": {
              "avg": 52368.444978499996,
              "p50": 53685.052888499995,
              "p90": 55061.153848099995,
              "p99": 55384.64992631
            },
            "time_to_first_token": {
              "avg": 7589.518373749999,
              "p50": 7690.770584,
              "p90": 10331.9980665,
              "p99": 10926.82953555
            }
          },
          "forward_passes_per_second": 30.369219127044477,
          "id": "r04",
          "output_tokens_per_forward_per_request": 3.081651954602774,
          "speculative": {
            "accept_length": {
              "max": 3.975,
              "mean": 3.136850649350649,
              "median": 3.00625,
              "min": 2.64375
            },
            "accept_rate": {
              "max": 0.7583333333333333,
              "mean": 0.6190584415584415,
              "median": 0.595,
              "min": 0.53875
            }
          },
          "synthetic_decode_tokens_per_second": 389.3073125125634
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 10.899286184004884,
              "p50": 10.934267947374845,
              "p90": 11.234305586862027,
              "p99": 11.33341356549939
            },
            "request_latency": {
              "avg": 52199.70420175,
              "p50": 52349.339681499994,
              "p90": 53653.0412993,
              "p99": 53690.519378029996
            },
            "time_to_first_token": {
              "avg": 7567.12727825,
              "p50": 7642.9162865,
              "p90": 10345.5713514,
              "p99": 10963.93798254
            }
          },
          "forward_passes_per_second": 31.22133363951746,
          "id": "r05",
          "output_tokens_per_forward_per_request": 3.067702552719201,
          "speculative": {
            "accept_length": {
              "max": 3.96875,
              "mean": 3.04066091954023,
              "median": 2.93125,
              "min": 2.575
            },
            "accept_rate": {
              "max": 0.9020833333333333,
              "mean": 0.626331417624521,
              "median": 0.6145833333333334,
              "min": 0.48
            }
          },
          "synthetic_decode_tokens_per_second": 385.7969511272832
        }
      ],
      "speculative": {
        "accept_length": {
          "max_per_run": {
            "count": 5,
            "max": 4.55,
            "mean": 3.95625,
            "median": 3.96875,
            "min": 3.41875,
            "sample_cv": 0.10186205157426655,
            "sample_stddev": 0.402991741540692
          },
          "mean_per_run": {
            "count": 5,
            "max": 3.2914383561643836,
            "mean": 3.067334740957917,
            "median": 3.04066091954023,
            "min": 2.8467147435897435,
            "sample_cv": 0.05321795439178651,
            "sample_stddev": 0.1632372803486407
          },
          "median_per_run": {
            "count": 5,
            "max": 3.09375,
            "mean": 2.96,
            "median": 2.95625,
            "min": 2.8125,
            "sample_cv": 0.03487470071294355,
            "sample_stddev": 0.10322911411031291
          },
          "min_per_run": {
            "count": 5,
            "max": 2.64375,
            "mean": 2.57,
            "median": 2.575,
            "min": 2.49375,
            "sample_cv": 0.029203138234004025,
            "sample_stddev": 0.07505206526139034
          }
        },
        "accept_rate": {
          "max_per_run": {
            "count": 5,
            "max": 0.9020833333333333,
            "mean": 0.7670833333333333,
            "median": 0.7583333333333333,
            "min": 0.675,
            "sample_cv": 0.10987547284079933,
            "sample_stddev": 0.08428364395829649
          },
          "mean_per_run": {
            "count": 5,
            "max": 0.6525114155251142,
            "mean": 0.6209722202387011,
            "median": 0.6190584415584415,
            "min": 0.5990331196581197,
            "sample_cv": 0.032989310978368505,
            "sample_stddev": 0.020485445682382447
          },
          "median_per_run": {
            "count": 5,
            "max": 0.645,
            "mean": 0.61675,
            "median": 0.6145833333333334,
            "min": 0.595,
            "sample_cv": 0.03240481872883446,
            "sample_stddev": 0.019985671951008654
          },
          "min_per_run": {
            "count": 5,
            "max": 0.5479166666666667,
            "mean": 0.5050833333333333,
            "median": 0.48375,
            "min": 0.475,
            "sample_cv": 0.06970040249510973,
            "sample_stddev": 0.035204511626905
          }
        }
      },
      "synthetic_decode_tokens_per_second": {
        "count": 5,
        "max": 409.87935669154285,
        "mean": 386.398364968282,
        "median": 385.7969511272832,
        "min": 366.48113143468447,
        "sample_cv": 0.040742529653376054,
        "sample_stddev": 15.742846842736252
      }
    }
  },
  "engine": "sglang",
  "interpretation": "same-process engineering regression signal; synthetic fixed-window output rate is not expected production, interactive, or application throughput and includes path-dependent speculative acceptance; repetitions are prompt-path subsamples, not independent deployment replicates",
  "mode": "repeat-c4",
  "prefill": {},
  "schema_version": "1.2"
}
