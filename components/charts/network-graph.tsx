/**
 * Network Graph Visualization
 * D3.js force-directed graph for relationship mapping
 */

'use client';

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { GlassCard, GlassCardHeader, GlassCardTitle, GlassCardContent } from '@/components/ui/glass-card';

interface Node {
  id: string;
  type: 'beneficiary' | 'shop' | 'inspector';
  label: string;
  riskScore?: number;
  x?: number;
  y?: number;
}

interface Link {
  source: string | Node;
  target: string | Node;
  relationshipType: string;
  strength: number;
}

interface NetworkGraphProps {
  nodes: Node[];
  links: Link[];
  width?: number;
  height?: number;
  title?: string;
}

export function NetworkGraph({
  nodes,
  links,
  width = 800,
  height = 600,
  title = 'Relationship Network'
}: NetworkGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height]);

    // Create container for zoom
    const g = svg.append('g');

    // Define arrow markers
    svg.append('defs').selectAll('marker')
      .data(['end'])
      .join('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#999');

    // Color scale for node types
    const colorScale = d3.scaleOrdinal<string>()
      .domain(['beneficiary', 'shop', 'inspector'])
      .range(['#3b82f6', '#10b981', '#f59e0b']);

    // Size scale based on risk score
    const sizeScale = d3.scaleLinear()
      .domain([0, 100])
      .range([8, 20]);

    // Force simulation
    const simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links)
        .id((d: any) => d.id)
        .distance((d: any) => 100 / (d.strength || 0.5))
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Create links
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', (d: any) => d.strength || 0.5)
      .attr('stroke-width', (d: any) => (d.strength || 0.5) * 3)
      .attr('marker-end', 'url(#arrow)');

    // Create nodes
    const node = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', (d: any) => sizeScale(d.riskScore || 50))
      .attr('fill', (d: any) => colorScale(d.type))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .call(drag(simulation) as any);

    // Add labels
    const label = g.append('g')
      .selectAll('text')
      .data(nodes)
      .join('text')
      .text((d: any) => d.label)
      .attr('font-size', 10)
      .attr('dx', 15)
      .attr('dy', 4)
      .attr('fill', '#374151')
      .style('pointer-events', 'none');

    // Tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'tooltip')
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('background-color', 'rgba(255, 255, 255, 0.95)')
      .style('backdrop-filter', 'blur(8px)')
      .style('padding', '12px')
      .style('border-radius', '8px')
      .style('box-shadow', '0 4px 6px rgba(0, 0, 0, 0.1)')
      .style('font-size', '12px')
      .style('z-index', '1000');

    node
      .on('mouseover', function (event, d: any) {
        tooltip.style('visibility', 'visible')
          .html(`
            <strong>${d.label}</strong><br/>
            Type: ${d.type}<br/>
            Risk Score: ${d.riskScore || 'N/A'}
          `);
      })
      .on('mousemove', function (event) {
        tooltip
          .style('top', (event.pageY - 10) + 'px')
          .style('left', (event.pageX + 10) + 'px');
      })
      .on('mouseout', function () {
        tooltip.style('visibility', 'hidden');
      });

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      label
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    });

    // Zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

    // Cleanup
    return () => {
      simulation.stop();
      tooltip.remove();
    };
  }, [nodes, links, width, height]);

  // Drag behavior
  function drag(simulation: d3.Simulation<any, undefined>) {
    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    return d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended);
  }

  return (
    <GlassCard>
      <GlassCardHeader>
        <GlassCardTitle>{title}</GlassCardTitle>
        <div className="flex gap-4 mt-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-500" />
            <span>Beneficiary</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-green-500" />
            <span>Shop</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-orange-500" />
            <span>Inspector</span>
          </div>
        </div>
      </GlassCardHeader>
      <GlassCardContent>
        <div className="overflow-hidden rounded-lg bg-gray-50">
          <svg ref={svgRef} />
        </div>
        <div className="mt-4 text-sm text-gray-600">
          <p>💡 <strong>Tip:</strong> Drag nodes to rearrange, scroll to zoom, larger nodes indicate higher risk scores</p>
        </div>
      </GlassCardContent>
    </GlassCard>
  );
}
