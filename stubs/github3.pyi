from typing import Iterator

class Repository:
    full_name: str
    stargazers: int

class RepositorySearchResult:
    repository: Repository

class GitHub:
    def search_repositories(self, query: str, sort: str=None) -> Iterator[RepositorySearchResult]: ...

def login(token: str=None) -> GitHub: ...
