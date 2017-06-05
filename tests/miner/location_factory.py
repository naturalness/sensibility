from sensibility.lexical_analysis import Position, Location


class LocationFactory:
    """
    Creates locations, incrementally.
    """
    def __init__(self, start: Position) -> None:
        self.current = start

    def across(self, width: int) -> Location:
        start = self.current
        self.current = Position(line=start.line, column=start.column + width)
        return Location(start=start, end=self.current)

    def until(self, end: Position) -> Location:
        start = self.current
        self.current = end
        return Location(start=start, end=end)

    def single(self) -> Location:
        return self.across(1)

    def newline(self) -> Location:
        result = self.single()
        self.next_line()
        return result

    def next_line(self, n: int=1) -> 'LocationFactory':
        self.current = Position(line=self.current.line + n, column=0)
        return self

    def space(self, n: int=1) -> 'LocationFactory':
        self.current = Position(line=self.current.line,
                                column=self.current.column + n)
        return self
