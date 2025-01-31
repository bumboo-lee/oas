class Order:
    def __init__(self,
                 order_no, order_date, decision_due_date,
                 model_name, processing_time, due_date,
                 revenue, risk):
        self.order_no = order_no
        self.order_date = order_date
        self.decision_due_date = decision_due_date
        self.model_name = model_name
        self.processing_time = processing_time
        self.due_date = due_date
        self.revenue = revenue
        self.risk = risk

        self.final_action = None
        self.decision_history = []
        self.start_time = None
        self.finish_time = None
        self.is_completed = False

    def __repr__(self):
        return (f"Order#{self.order_no}(OD={self.order_date},"
                f" DDD={self.decision_due_date}, model={self.model_name})")
