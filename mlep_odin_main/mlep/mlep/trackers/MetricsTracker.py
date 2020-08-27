

class MetricsTracker:
    def __init__(self,):
        self.METRICS={}
        self.METRICS["all_errors"] = []
        self.METRICS['classification'] = None
        self.METRICS['error'] = None
        self.METRICS['ensembleRaw'] = []
        self.METRICS['ensembleWeighted'] = []
        self.METRICS['ensembleError'] = []


    def updateMetrics(self, classification, error, ensembleError, ensembleRaw, ensembleWeighted):
        self.METRICS["all_errors"].append(error)
        self.METRICS['classification'] = classification
        self.METRICS['error'] = error
        self.METRICS['ensembleRaw'] = ensembleRaw
        self.METRICS['ensembleWeighted'] = ensembleWeighted
        self.METRICS['ensembleError'] = ensembleError

    def get(self, key):
        return self.METRICS[key]