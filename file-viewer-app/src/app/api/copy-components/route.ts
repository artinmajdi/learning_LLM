import { NextResponse } from 'next/server';
import { copyComponentsToProject } from '@/utils/copyComponents';

export async function GET() {
  try {
    const componentNames = await copyComponentsToProject();
    return NextResponse.json({ success: true, componentNames });
  } catch (error) {
    console.error('Error in copy-components API route:', error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
