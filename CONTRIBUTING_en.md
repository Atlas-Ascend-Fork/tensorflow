# Contributing Guidelines

This project welcomes developers to experience and contribute. Before participating in community contributions, refer to [cann-community](https://gitcode.com/cann/community) for the code of conduct, sign the CLA agreement, and learn about the source repository contribution process. That repository provides detailed prerequisites for contributing to the CANN open-source project, including but not limited to:

1. How to submit a PR
2. GitCode workflow
3. Pipeline trigger commands
4. Code review
5. Other considerations

For details, refer to [cann-community](https://gitcode.com/cann/community).

In addition, when preparing local code and submitting PRs, developers should pay close attention to the following points:

1. When submitting a PR, fill in the business background, purpose, and solution of the PR according to the PR template.
2. If your change is not a simple bug fix but involves new features, new interfaces, new configuration parameters, or modified code flows, discuss the solution through an Issue first to avoid your code being rejected. If you are unsure whether the change qualifies as a "simple bug fix," you can also submit an Issue for discussion.
3. When submitting a PR, ensure your code follows the project coding standards. Refer to Google's [Open Source Coding Standards](https://google.github.io/styleguide/), including but not limited to:
   - Code formatting
   - Comment standards
   - Variable naming conventions
   - Function naming conventions
   - Class naming conventions
   - Interface naming conventions
   - Configuration parameter naming conventions
   - Code flow standards
4. When submitting a PR, if there are multiple invalid commits, rebase to squash them into one before submitting. This maintains code conciseness and readability. Refer to [git rebase](https://git-scm.com/docs/git-rebase). Additionally, commit messages must follow the project coding standards and clearly describe the intent and content of the change. The format is: `<type>: <short description>`. For example:

| Type     | Description                          | Example                              |
|----------|--------------------------------------|--------------------------------------|
| feat     | New feature                          | feat: add user registration          |
| fix      | Bug fix                              | fix: resolve login session expiry    |
| docs     | Documentation update                 | docs: update API usage guide         |
| style    | Code style change (no logic change)  | style: adjust code indentation       |
| refactor | Refactoring (not new feature/fix)    | refactor: optimize user service class|
| perf     | Performance improvement              | perf: reduce database queries        |
| test     | Test-related                         | test: add login unit tests           |
| chore    | Build/toolchain change               | chore: update webpack config         |
| ci       | CI configuration                     | ci: add automated test pipeline      |

Developer contribution scenarios mainly include:

- Bug Fix

  If you find a bug in this project and want to fix it, create an Issue to report and track it.

  Follow the [Submit Issue/Handle Issue Tasks](https://gitcode.com/cann/community) guide to create a `Bug-Report|Defect Feedback` type Issue describing the bug. Then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself.

- New Feature Contribution

  If you find missing functionality in this project and want to add it, create an Issue to report and track it.

  Follow the [Submit Issue/Handle Issue Tasks](https://gitcode.com/cann/community) guide to create a `Requirement|Feature Suggestion` type Issue describing the new feature and providing your design solution. Then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for tracking and implementation.

- Documentation Correction

  If you find documentation errors in this project, create an Issue to report and fix them.

  Follow the [Submit Issue/Handle Issue Tasks](https://gitcode.com/cann/community) guide to create a `Documentation|Documentation Feedback` type Issue pointing out the documentation issue. Then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself to correct the documentation.

- Help Resolve Others' Issues

  If you have suitable solutions for issues others encounter in the community, leave comments in the Issue to help others resolve problems and improve usability together.

  If the Issue requires code changes, enter "/assign" or "/assign @yourself" in the Issue comment box to assign the Issue to yourself and track the resolution.
