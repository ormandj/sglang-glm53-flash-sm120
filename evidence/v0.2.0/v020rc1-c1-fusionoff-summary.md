{
  "build_id": "v0.2.0-rc.1-c1-fusionoff-450k",
  "decode": {
    "c1": {
      "client_latency_ms": {
        "inter_token_latency": {
          "avg_per_run": {
            "count": 5,
            "max": 5.847530305494506,
            "mean": 5.095239662857143,
            "median": 5.3786144131868125,
            "min": 3.9266065484737482,
            "sample_cv": 0.1456059807921375,
            "sample_stddev": 0.7418973684813143
          },
          "p50_per_run": {
            "count": 5,
            "max": 5.847530305494506,
            "mean": 5.095239662857143,
            "median": 5.3786144131868125,
            "min": 3.9266065484737482,
            "sample_cv": 0.1456059807921375,
            "sample_stddev": 0.7418973684813143
          },
          "p90_per_run": {
            "count": 5,
            "max": 5.847530305494506,
            "mean": 5.095239662857143,
            "median": 5.3786144131868125,
            "min": 3.9266065484737482,
            "sample_cv": 0.1456059807921375,
            "sample_stddev": 0.7418973684813143
          },
          "p99_per_run": {
            "count": 5,
            "max": 5.847530305494506,
            "mean": 5.095239662857143,
            "median": 5.3786144131868125,
            "min": 3.9266065484737482,
            "sample_cv": 0.1456059807921375,
            "sample_stddev": 0.7418973684813143
          }
        },
        "request_latency": {
          "avg_per_run": {
            "count": 5,
            "max": 27092.515487,
            "mean": 23876.2219602,
            "median": 24859.560826999998,
            "min": 18902.499918999998,
            "sample_cv": 0.13142850651569096,
            "sample_stddev": 3138.016193466229
          },
          "p50_per_run": {
            "count": 5,
            "max": 27092.515487,
            "mean": 23876.2219602,
            "median": 24859.560826999998,
            "min": 18902.499918999998,
            "sample_cv": 0.13142850651569096,
            "sample_stddev": 3138.016193466229
          },
          "p90_per_run": {
            "count": 5,
            "max": 27092.515487,
            "mean": 23876.2219602,
            "median": 24859.560826999998,
            "min": 18902.499918999998,
            "sample_cv": 0.13142850651569096,
            "sample_stddev": 3138.016193466229
          },
          "p99_per_run": {
            "count": 5,
            "max": 27092.515487,
            "mean": 23876.2219602,
            "median": 24859.560826999998,
            "min": 18902.499918999998,
            "sample_cv": 0.13142850651569096,
            "sample_stddev": 3138.016193466229
          }
        },
        "time_to_first_token": {
          "avg_per_run": {
            "count": 5,
            "max": 3146.878886,
            "mean": 3011.2155408,
            "median": 3124.648373,
            "min": 2823.0461029999997,
            "sample_cv": 0.055452526333165664,
            "sample_stddev": 166.97950907104968
          },
          "p50_per_run": {
            "count": 5,
            "max": 3146.878886,
            "mean": 3011.2155408,
            "median": 3124.648373,
            "min": 2823.0461029999997,
            "sample_cv": 0.055452526333165664,
            "sample_stddev": 166.97950907104968
          },
          "p90_per_run": {
            "count": 5,
            "max": 3146.878886,
            "mean": 3011.2155408,
            "median": 3124.648373,
            "min": 2823.0461029999997,
            "sample_cv": 0.055452526333165664,
            "sample_stddev": 166.97950907104968
          },
          "p99_per_run": {
            "count": 5,
            "max": 3146.878886,
            "mean": 3011.2155408,
            "median": 3124.648373,
            "min": 2823.0461029999997,
            "sample_cv": 0.055452526333165664,
            "sample_stddev": 166.97950907104968
          }
        }
      },
      "engine_forward_passes_per_second": {
        "count": 5,
        "max": 61.07476987546551,
        "mean": 58.13050817860009,
        "median": 58.42867579817825,
        "min": 52.16753613376154,
        "sample_cv": 0.062158089121979544,
        "sample_stddev": 3.613281308071385
      },
      "output_tokens_per_forward_per_request": {
        "count": 5,
        "max": 5.511406844106464,
        "mean": 3.6473919370634134,
        "median": 3.117021276595745,
        "min": 2.751592356687898,
        "sample_cv": 0.30316130021291343,
        "sample_stddev": 1.1057480820262413
      },
      "repetitions": [
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 5.3786144131868125,
              "p50": 5.3786144131868125,
              "p90": 5.3786144131868125,
              "p99": 5.3786144131868125
            },
            "request_latency": {
              "avg": 24859.560826999998,
              "p50": 24859.560826999998,
              "p90": 24859.560826999998,
              "p99": 24859.560826999998
            },
            "time_to_first_token": {
              "avg": 2834.1348049999997,
              "p50": 2834.1348049999997,
              "p90": 2834.1348049999997,
              "p99": 2834.1348049999997
            }
          },
          "forward_passes_per_second": 60.941281938521385,
          "id": "r01",
          "output_tokens_per_forward_per_request": 3.117021276595745,
          "speculative": {
            "accept_length": {
              "max": 5.925,
              "mean": 3.0421875000000003,
              "median": 2.575,
              "min": 2.325
            },
            "accept_rate": {
              "max": 0.985,
              "mean": 0.5522569444444444,
              "median": 0.5,
              "min": 0.315
            }
          },
          "synthetic_decode_tokens_per_second": 177.13283536965392
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 3.9266065484737482,
              "p50": 3.9266065484737482,
              "p90": 3.9266065484737482,
              "p99": 3.9266065484737482
            },
            "request_latency": {
              "avg": 18902.499918999998,
              "p50": 18902.499918999998,
              "p90": 18902.499918999998,
              "p99": 18902.499918999998
            },
            "time_to_first_token": {
              "avg": 2823.0461029999997,
              "p50": 2823.0461029999997,
              "p90": 2823.0461029999997,
              "p99": 2823.0461029999997
            }
          },
          "forward_passes_per_second": 52.16753613376154,
          "id": "r02",
          "output_tokens_per_forward_per_request": 5.511406844106464,
          "speculative": {
            "accept_length": {
              "max": 6.0,
              "mean": 5.266935483870968,
              "median": 5.925,
              "min": 2.475
            },
            "accept_rate": {
              "max": 1.0,
              "mean": 0.8901612903225806,
              "median": 0.985,
              "min": 0.49166666666666664
            }
          },
          "synthetic_decode_tokens_per_second": 297.0216847940745
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 5.847530305494506,
              "p50": 5.847530305494506,
              "p90": 5.847530305494506,
              "p99": 5.847530305494506
            },
            "request_latency": {
              "avg": 27092.515487,
              "p50": 27092.515487,
              "p90": 27092.515487,
              "p99": 27092.515487
            },
            "time_to_first_token": {
              "avg": 3146.878886,
              "p50": 3146.878886,
              "p90": 3146.878886,
              "p99": 3146.878886
            }
          },
          "forward_passes_per_second": 61.07476987546551,
          "id": "r03",
          "output_tokens_per_forward_per_request": 2.751592356687898,
          "speculative": {
            "accept_length": {
              "max": 3.375,
              "mean": 2.748636363636364,
              "median": 2.65,
              "min": 2.3
            },
            "accept_rate": {
              "max": 0.75,
              "mean": 0.5549999999999999,
              "median": 0.5333333333333333,
              "min": 0.325
            }
          },
          "synthetic_decode_tokens_per_second": 167.20424161494023
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 5.4624399995115995,
              "p50": 5.4624399995115995,
              "p90": 5.4624399995115995,
              "p99": 5.4624399995115995
            },
            "request_latency": {
              "avg": 25496.061335,
              "p50": 25496.061335,
              "p90": 25496.061335,
              "p99": 25496.061335
            },
            "time_to_first_token": {
              "avg": 3127.369537,
              "p50": 3127.369537,
              "p90": 3127.369537,
              "p99": 3127.369537
            }
          },
          "forward_passes_per_second": 58.42867579817825,
          "id": "r04",
          "output_tokens_per_forward_per_request": 3.08455114822547,
          "speculative": {
            "accept_length": {
              "max": 5.375,
              "mean": 3.12,
              "median": 2.8625,
              "min": 2.375
            },
            "accept_rate": {
              "max": 0.875,
              "mean": 0.6102,
              "median": 0.6083333333333333,
              "min": 0.41
            }
          },
          "synthetic_decode_tokens_per_second": 181.17533219233763
        },
        {
          "client_latency_ms": {
            "inter_token_latency": {
              "avg": 4.861007047619047,
              "p50": 4.861007047619047,
              "p90": 4.861007047619047,
              "p99": 4.861007047619047
            },
            "request_latency": {
              "avg": 23030.472233,
              "p50": 23030.472233,
              "p90": 23030.472233,
              "p99": 23030.472233
            },
            "time_to_first_token": {
              "avg": 3124.648373,
              "p50": 3124.648373,
              "p90": 3124.648373,
              "p99": 3124.648373
            }
          },
          "forward_passes_per_second": 58.04027714707374,
          "id": "r05",
          "output_tokens_per_forward_per_request": 3.7723880597014925,
          "speculative": {
            "accept_length": {
              "max": 6.0,
              "mean": 3.7802325581395353,
              "median": 3.125,
              "min": 2.2
            },
            "accept_rate": {
              "max": 1.0,
              "mean": 0.6904651162790698,
              "median": 0.6833333333333333,
              "min": 0.395
            }
          },
          "synthetic_decode_tokens_per_second": 210.27821569710872
        }
      ],
      "speculative": {
        "accept_length": {
          "max_per_run": {
            "count": 5,
            "max": 6.0,
            "mean": 5.335,
            "median": 5.925,
            "min": 3.375,
            "sample_cv": 0.2111473777272517,
            "sample_stddev": 1.1264712601748879
          },
          "mean_per_run": {
            "count": 5,
            "max": 5.266935483870968,
            "mean": 3.5915983811293737,
            "median": 3.12,
            "min": 2.748636363636364,
            "sample_cv": 0.28110084883632336,
            "sample_stddev": 1.0096013536146318
          },
          "median_per_run": {
            "count": 5,
            "max": 5.925,
            "mean": 3.4274999999999998,
            "median": 2.8625,
            "min": 2.575,
            "sample_cv": 0.41207890268973374,
            "sample_stddev": 1.4124004389690623
          },
          "min_per_run": {
            "count": 5,
            "max": 2.475,
            "mean": 2.335,
            "median": 2.325,
            "min": 2.2,
            "sample_cv": 0.04322618680553821,
            "sample_stddev": 0.10093314619093173
          }
        },
        "accept_rate": {
          "max_per_run": {
            "count": 5,
            "max": 1.0,
            "mean": 0.922,
            "median": 0.985,
            "min": 0.75,
            "sample_cv": 0.11872516936219414,
            "sample_stddev": 0.109464606151943
          },
          "mean_per_run": {
            "count": 5,
            "max": 0.8901612903225806,
            "mean": 0.6596166702092189,
            "median": 0.6102,
            "min": 0.5522569444444444,
            "sample_cv": 0.21307561092939142,
            "sample_stddev": 0.1405482249840402
          },
          "median_per_run": {
            "count": 5,
            "max": 0.985,
            "mean": 0.662,
            "median": 0.6083333333333333,
            "min": 0.5,
            "sample_cv": 0.29297283548412517,
            "sample_stddev": 0.19394801709049087
          },
          "min_per_run": {
            "count": 5,
            "max": 0.49166666666666664,
            "mean": 0.3873333333333333,
            "median": 0.395,
            "min": 0.315,
            "sample_cv": 0.18516585982315031,
            "sample_stddev": 0.07172090970483355
          }
        }
      },
      "synthetic_decode_tokens_per_second": {
        "count": 5,
        "max": 297.0216847940745,
        "mean": 206.56246193362298,
        "median": 181.17533219233763,
        "min": 167.20424161494023,
        "sample_cv": 0.25681330793901397,
        "sample_stddev": 53.047989145200376
      }
    }
  },
  "engine": "sglang",
  "interpretation": "same-process engineering regression signal; synthetic fixed-window output rate is not expected production, interactive, or application throughput and includes path-dependent speculative acceptance; repetitions are prompt-path subsamples, not independent deployment replicates",
  "mode": "glm-c1",
  "prefill": {},
  "schema_version": "1.2"
}
