#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "CycleTimer.h"
#include "bfs.h"
#include "graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->alloc_count = count;
    list->present = (int*)malloc(sizeof(int) * list->alloc_count);
    vertex_set_clear(list);
}

// void bfs_hybrid(graph* graph, solution* sol)
// {
//     vertex_set list1;
//     vertex_set list2;
//     vertex_set_init(&list1, graph->num_nodes);
//     vertex_set_init(&list2, graph->num_nodes);

//     vertex_set* frontier = &list1;
//     vertex_set* new_frontier = &list2;

//     // initialize all nodes to NOT_VISITED
//     for (int i=0; i<graph->num_nodes; i++)
//         sol->distances[i] = NOT_VISITED_MARKER;

//     // setup frontier with the root node
//     frontier->present[frontier->count++] = ROOT_NODE_ID;
//     sol->distances[ROOT_NODE_ID] = 0;

//     int step = 0;
//     while (frontier->count != 0) {

// #ifdef DEBUG
//         double start_time = CycleTimer::currentSeconds();
// #endif

//         vertex_set_clear(new_frontier);

//         if ()
//         bottom_up_step(graph, frontier, new_frontier, sol->distances, step++);

// #ifdef DEBUG
//         double end_time = CycleTimer::currentSeconds();
//         printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
// #endif

//         // swap pointers
//         vertex_set* tmp = frontier;
//         frontier = new_frontier;
//         new_frontier = tmp;
//     }

//     free(frontier->present);
//     free(new_frontier->present);
// }


// We note that in step s of the BFS, everything that should be added to the new
// frontier has a neighbor with distance s (and everything in the frontier has
// distance s), and no neighbors with distance less than s

int bottom_up_step(
    graph* g,
    char* frontier,
    char* new_frontier,
    int* distances,
    int step)
{
    int frontier_full = 0;
    static int local_frontier_full = 0;

    #pragma omp threadprivate(local_frontier_full)

    #pragma omp parallel shared(frontier_full)
    {
        local_frontier_full = 0;

        #pragma omp for
        for (int node = 0; node < g->num_nodes; node++) {
            if (distances[node] == NOT_VISITED_MARKER) {
                int start_edge = g->incoming_starts[node];
                int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[node + 1];

                for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int distance = distances[g->incoming_edges[neighbor]];

                    if (distance == step) {
                        distances[node] = distance + 1;
                        new_frontier[node] = 1;
                        local_frontier_full = 1;
                        // new_frontier->present[__sync_fetch_and_add(&(new_frontier->count), 1)] = node;
                        break;
                    }
                }
            }
        }

        #pragma omp atomic
        frontier_full += local_frontier_full;
    }


    return frontier_full;
}

void bfs_bottom_up(graph* graph, solution* sol)
{
    // 15-418/618 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    char* frontier = new char[graph->num_nodes], *new_frontier = new char[graph->num_nodes];

    memset(frontier, 0, sizeof(char) * graph->num_nodes);
    memset(new_frontier, 0, sizeof(char) * graph->num_nodes);


    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    // frontier->present[frontier->count++] = ROOT_NODE_ID;
    frontier[ROOT_NODE_ID] = 1;
    sol->distances[ROOT_NODE_ID] = 0;

    int step = 0, frontier_full = 1;
    while (frontier_full != 0) {

#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        // memset(new_frontier->front(), 0, sizeof(char) * graph->num_nodes);
        // new_frontier->clear();
        // new_frontier->resize(graph->num_nodes, 0);
        memset(new_frontier, 0, sizeof(char) * graph->num_nodes);

        frontier_full = bottom_up_step(graph, frontier, new_frontier, sol->distances, step++);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", 1, end_time - start_time);
#endif

        // swap pointers
        char* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}


// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    graph* g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{

    #pragma omp parallel for
    for (int i=0; i<frontier->count; i++) {

        int node = frontier->present[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes-1) ? g->num_edges : g->outgoing_starts[node+1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                distances[outgoing] = distances[node] + 1;
                new_frontier->present[__sync_fetch_and_add(&(new_frontier->count), 1)] = outgoing;
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(graph* graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->present[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(frontier->present);
    free(new_frontier->present);
}
