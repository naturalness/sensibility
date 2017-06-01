#include <stdio.h>
#include <stddef.h>

#include "errcode.h"
#include "Python.h"

/*=============== COPY-PASTED FROM cpython/Parser/tokenize.h ===============*/

/*
 Python License:

1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"), and
   the Individual or Organization ("Licensee") accessing and otherwise using Python
   3.6.1 software in source or binary form and its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
   grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
   analyze, test, perform and/or display publicly, prepare derivative works,
   distribute, and otherwise use Python 3.6.1 alone or in any derivative
   version, provided, however, that PSF's License Agreement and PSF's notice of
   copyright, i.e., "Copyright © 2001-2017 Python Software Foundation; All Rights
   Reserved" are retained in Python 3.6.1 alone or in any derivative version
   prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on or
   incorporates Python 3.6.1 or any part thereof, and wants to make the
   derivative work available to others as provided herein, then Licensee hereby
   agrees to include in any such work a brief summary of the changes made to Python
   3.6.1.

4. PSF is making Python 3.6.1 available to Licensee on an "AS IS" basis.
   PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED.  BY WAY OF
   EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR
   WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE
   USE OF PYTHON 3.6.1 WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON 3.6.1
   FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
   MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON 3.6.1, OR ANY DERIVATIVE
   THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material breach of
   its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any relationship
   of agency, partnership, or joint venture between PSF and Licensee.  This License
   Agreement does not grant permission to use PSF trademarks or trade name in a
   trademark sense to endorse or promote products or services of Licensee, or any
   third party.

8. By copying, installing or otherwise using Python 3.6.1, Licensee agrees
   to be bound by the terms and conditions of this License Agreement.
*/

#include "object.h"
#include "token.h"

#define MAXINDENT 100   /* Max indentation level */

enum decoding_state {
    STATE_INIT,
    STATE_RAW,
    STATE_NORMAL        /* have a codec associated with input */
};

/* Tokenizer state */
struct tok_state {
    /* Input state; buf <= cur <= inp <= end */
    /* NB an entire line is held in the buffer */
    char *buf;          /* Input buffer, or NULL; malloc'ed if fp != NULL */
    char *cur;          /* Next character in buffer */
    char *inp;          /* End of data in buffer */
    char *end;          /* End of input buffer if buf != NULL */
    char *start;        /* Start of current token if not NULL */
    int done;           /* E_OK normally, E_EOF at EOF, otherwise error code */
    /* NB If done != E_OK, cur must be == inp!!! */
    FILE *fp;           /* Rest of input; NULL if tokenizing a string */
    int tabsize;        /* Tab spacing */
    int indent;         /* Current indentation index */
    int indstack[MAXINDENT];            /* Stack of indents */
    int atbol;          /* Nonzero if at begin of new line */
    int pendin;         /* Pending indents (if > 0) or dedents (if < 0) */
    char *prompt, *nextprompt;          /* For interactive prompting */
    int lineno;         /* Current line number */
    int level;          /* () [] {} Parentheses nesting level */
            /* Used to allow free continuations inside them */
    /* Stuff for checking on different tab sizes */
    const char *filename;   /* encoded to the filesystem encoding */
    int altwarning;     /* Issue warning if alternate tabs don't match */
    int alterror;       /* Issue error if alternate tabs don't match */
    int alttabsize;     /* Alternate tab spacing */
    int altindstack[MAXINDENT];         /* Stack of alternate indents */
    /* Stuff for PEP 0263 */
    enum decoding_state decoding_state;
    int decoding_erred;         /* whether erred in decoding  */
    int read_coding_spec;       /* whether 'coding:...' has been read  */
    char *encoding;         /* Source encoding. */
    int cont_line;          /* whether we are in a continuation line. */
    const char* line_start;     /* pointer to start of current line */

    const char* enc;        /* Encoding for the current str. */
    const char* str;
    const char* input; /* Tokenizer's newline translated copy of the string. */
};

struct tok_state *PyTokenizer_FromString(const char *, int);
struct tok_state *PyTokenizer_FromUTF8(const char *, int);
struct tok_state *PyTokenizer_FromFile(FILE *, char*,
                                              char *, char *);
void PyTokenizer_Free(struct tok_state *);
int PyTokenizer_Get(struct tok_state *, char **, char **);
/*============================= END COPY-PASTE =============================*/


/* There's a table of token names. Use it! */
extern const char *_PyParser_TokenNames[];

/* Here is some example CPython code. */
static char sample[] =
    "#!/usr/bin/env python\n"
    "# -*- coding: latin-1 -*-\n"
    "\n"
    "print('Good morning \241por la ma\361ana!')\n";


int main(void) {
    struct tok_state *state;
    state = PyTokenizer_FromString(sample, 0);
    char *a, *b;

    for (;;) {
        int tok = PyTokenizer_Get(state, &a, &b);
        ptrdiff_t len = b - a;

        printf("%s\t", _PyParser_TokenNames[tok]);
        if (len)
            fwrite(a, sizeof(char), len, stdout);
        putchar('\n');

        if (tok == ENDMARKER) break;
    }

    PyTokenizer_Free(state);
    return 0;
}
