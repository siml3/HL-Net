


### Performance
|      case                  |    Type      |  lr schd  |  im/gpu | bbox AP | mask AP |
|----------------------------|:------------:|:---------:|:-------:|:-------:|:-------:|
|   R-50-FPN, GN (paper)     | finetune     |    2x     |   2     |   40.3  |  35.7   |
|   R-50-FPN, GN (implement) | finetune     |    2x     |   2     |   40.2  |  36.0   |
|   R-50-FPN, GN (paper)     | from scratch |    3x     |   2     |   39.5  |  35.2   |
|   R-50-FPN, GN (implement) | from scratch |    3x     |   2     |   38.9  |  35.1   |
