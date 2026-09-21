"""Step-by-step tasks written as generators: wait, hold until a problem clears, run one at a time."""
class TaskError(Exception):
    """A task stopped for a reason worth telling the user."""


def elapsed(clock, since):
    return (clock.now() - since).nanoseconds * 1e-9


def sleep(clock, seconds):
    start = clock.now()
    while elapsed(clock, start) < seconds:
        yield


def hold(clock, problem, duration, timeout=None, what='timed out'):
    """Wait until `problem()` has returned '' for `duration` seconds in a row, or raise once `timeout` passes."""
    start, clean_since, reason = clock.now(), None, ''
    while True:
        current = problem()
        if current:
            clean_since, reason = None, current
        elif clean_since is None:
            clean_since = clock.now()

        if clean_since is not None and elapsed(clock, clean_since) >= duration:
            return
        if timeout is not None and elapsed(clock, start) > timeout:
            raise TaskError(f'{what}: {reason}')
        yield


class Periodic:
    """Calls `action` when polled, at most once per `period` seconds; the first poll only starts the clock."""

    def __init__(self, clock, period, action):
        self.clock, self.period, self.action = clock, period, action
        self.last = clock.now()

    def __call__(self):
        if elapsed(self.clock, self.last) >= self.period:
            self.last = self.clock.now()
            self.action()


class Tasks:
    """Runs one generator at a time, a step per timer tick; a task yields to wait for the next tick."""

    def __init__(self, node, period=0.02):
        self.node = node
        self.task = None
        node.create_timer(period, self.step)

    @property
    def busy(self):
        return self.task is not None

    def start(self, task):
        """Run `task` in place of any current one; returns why it stopped at once, or '' if it did not."""
        self.cancel()
        self.task = task
        return self.step()

    def cancel(self):
        if self.task is not None:
            self.task.close()
            self.task = None

    def step(self):
        if self.task is None:
            return ''
        try:
            next(self.task)
        except StopIteration:
            self.task = None
        except TaskError as err:
            self.task = None
            self.node.get_logger().error(str(err))
            return str(err)
        return ''
