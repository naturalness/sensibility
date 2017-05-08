from typing import Iterator, Dict

class Repository:
    full_name: str
    stargazers: int

class RepositorySearchResult:
    repository: Repository

class GitHub:
    def search_repositories(self, query: str, sort: str=None) -> Iterator[RepositorySearchResult]: ...
    def rate_limit(self) -> Dict[str, Dict[str, Dict[str, int]]]: ...

def login(token: str=None) -> GitHub: ...
