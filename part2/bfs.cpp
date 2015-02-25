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

inline int get_start_edge(int vertex, graph* g) {
    return g->incoming_starts[vertex];
}

inline int get_end_edge(int vertex, graph* g) {
    return (vertex == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[vertex + 1];
}

// We note that in step s of the BFS, everything that should be added to the new
// frontier has a neighbor with distance s (and everything in the frontier has
// distance s), and no neighbors with distance less than s

void bottom_up_step(
    graph* g,
    char* frontier,
    char* new_frontier,
    int* distances,
    int step,
    graph_info* info)
{
    info->frontier_edges = 0;
    info->frontier_vertices = 0;

    static size_t local_frontier_size = 0, local_explored_edges = 0, local_frontier_edges = 0;

    #pragma omp threadprivate(local_frontier_size, local_frontier_edges, local_explored_edges)

    #pragma omp parallel
    {
        local_frontier_size = 0;
        local_frontier_edges = 0;
        local_explored_edges = 0;

        #pragma omp for
        for (int node = 0; node < g->num_nodes; node++) {
            if (distances[node] == NOT_VISITED_MARKER) {
                int start_edge = get_start_edge(node, g);
                int end_edge = get_end_edge(node, g);

                for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int distance = distances[g->incoming_edges[neighbor]];

                    if (distance == step) {
                        distances[node] = distance + 1;
                        new_frontier[node] = 1;
                        local_frontier_size++;

                        break;
                    }
                }
                local_explored_edges += (end_edge - start_edge);
            }
        }

        #pragma omp critical
        {
            info->frontier_vertices += local_frontier_size;
            info->unexplored_edges -= local_explored_edges;
            info->frontier_edges += local_explored_edges;
        }
    }
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

    int step = 0;
    graph_info info;

    info.unexplored_edges = graph->num_edges;
    info.frontier_vertices = 1;
    info.frontier_edges = get_end_edge(ROOT_NODE_ID, graph) - get_start_edge(ROOT_NODE_ID, graph);

    while (info.frontier_vertices != 0) {

#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        memset(new_frontier, 0, sizeof(char) * graph->num_nodes);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, step++, &info);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", 1, end_time - start_time);
#endif

        // swap pointers
        char* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(frontier);
    free(new_frontier);
}


// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    graph* g,
    char* frontier,
    char* new_frontier,
    int* distances,
    int old_frontier_size,
    graph_info* info)
{
    info->frontier_edges = 0;
    info->frontier_vertices = 0;

    static size_t local_frontier_size = 0, local_frontier_edges = 0, local_explored_edges = 0;

    #pragma omp threadprivate(local_frontier_size, local_frontier_edges, local_explored_edges)

    #pragma omp parallel
    {
        local_frontier_size = 0;
        local_frontier_edges = 0;
        local_explored_edges = 0;
        #pragma omp for

        for (int i=0; i<g->num_nodes; i++) {
            if (frontier[i]) {
                int node = i;

                int start_edge = get_start_edge(node, g);
                int end_edge = get_end_edge(node, g);

                // attempt to add all neighbors to the new frontier
                for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                    int outgoing = g->outgoing_edges[neighbor];

                    if (distances[outgoing] == NOT_VISITED_MARKER) {
                        distances[outgoing] = distances[node] + 1;
                        new_frontier[outgoing] = 1;
                        local_frontier_size++;
                        local_frontier_edges += (get_end_edge(node, g) - get_start_edge(node, g));
                    }
                }
                local_explored_edges += (end_edge - start_edge);
            }
        }

        #pragma omp critical
        {
            info->frontier_vertices += local_frontier_size;
            info->frontier_edges += local_frontier_edges;
            info->unexplored_edges -= local_explored_edges;
        }
    }

}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(graph* graph, solution* sol) {
    char* frontier = new char[graph->num_nodes], *new_frontier = new char[graph->num_nodes];

    memset(frontier, 0, sizeof(char) * graph->num_nodes);
    memset(new_frontier, 0, sizeof(char) * graph->num_nodes);

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier[ROOT_NODE_ID] = 1;
    sol->distances[ROOT_NODE_ID] = 0;

    graph_info info;

    info.unexplored_edges = graph->num_edges;
    info.frontier_vertices = 1;
    info.frontier_edges = get_end_edge(ROOT_NODE_ID, graph) - get_start_edge(ROOT_NODE_ID, graph);

    while (info.frontier_vertices != 0) {

#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        memset(new_frontier, 0, sizeof(char) * graph->num_nodes);

        top_down_step(graph, frontier, new_frontier, sol->distances, info.frontier_vertices, &info);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", info.frontier_vertices, end_time - start_time);
#endif

        // swap pointers
        char* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(frontier);
    free(new_frontier);
}

#define ALPHA 14

void bfs_hybrid(graph* graph, solution* sol)
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

    int step = 0;
    graph_info info;

    info.unexplored_edges = graph->num_edges;
    info.frontier_vertices = 1;
    info.frontier_edges = get_end_edge(ROOT_NODE_ID, graph) - get_start_edge(ROOT_NODE_ID, graph);

    while (info.frontier_vertices != 0) {

#ifdef DEBUG_HYBRID
        double start_time = CycleTimer::currentSeconds();
#endif

        memset(new_frontier, 0, sizeof(char) * graph->num_nodes);

        if (info.frontier_edges > info.unexplored_edges / ALPHA) {
            bottom_up_step(graph, frontier, new_frontier, sol->distances, step, &info);
        }
        else {
            top_down_step(graph, frontier, new_frontier, sol->distances, info.frontier_vertices, &info);
        }

        step++;


#ifdef DEBUG_HYBRID
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", info.frontier_vertices, end_time - start_time);
#endif

        // swap pointers
        char* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(frontier);
    free(new_frontier);
}

